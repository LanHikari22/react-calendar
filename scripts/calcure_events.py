import sys
import csv
import json
import hashlib
import time
import os
import pipe
import pandas as pd
from typing import List, Any, Tuple, Optional
from datetime import datetime as dtdt
from datetime import timedelta

# TODO: Update this path for the one on your system. (Need to be made externally configurable)
CALCURE_EVENTS_CSV_PATH = '/root/.task/schedule/events.csv'
MARKERS_JSON_PATH = 'public/markers.json'

class CalcureEvent:
    def __init__(self, _id: str, begin_dt: dtdt, interval: timedelta, end_dt: dtdt, 
                 event_type: str, gcode: str, uuid: str, subtask_uuid: str, proj: str, desc: str,
                 repeat_count: int, repeat_state: str, priority: str):
        self._id = _id
        self.begin_dt = begin_dt
        self.interval = interval
        self.end_dt = end_dt
        self.event_type = event_type
        self.gcode = gcode
        self.uuid = uuid
        self.subtask_uuid = subtask_uuid
        self.proj = proj
        self.desc = desc
        self.repeat_count = repeat_count
        self.repeat_state = repeat_state
        self.priority = priority

    @staticmethod
    def read_csv(row: List[str]) -> 'CalcureEvent':
        # ['217', '2023', '7', '7', '1340:0021 EVNT DRV1 afd865ee drvlic Request Driving car on exam time', '1', 'once', 'unimportant']
        assert len(row) == 8
        _id = row[0]
        _year = row[1]
        _month = row[2]
        _day = row[3]
        _desc = row[4]
        _repeat_count = row[5]
        _repeat_state = row[6]
        _priority = row[7]

        # Let's improve on this by using begin and end dates. We have enough information from the description which is guaranteed to have
        # begin_time:interval event_type gcode uuid project desc
        desc_tokens = _desc.split(' ')
        assert len(desc_tokens) > 5
        begin_time_and_intervals = desc_tokens[0] # hhmm:hhmm, 24-hour
        begin_time_s = begin_time_and_intervals.split(':')[0]
        begin_time_s_hh = int(begin_time_s[:2]) % 24
        begin_time_s_mm = begin_time_s[2:]
        interval_time_s = begin_time_and_intervals.split(':')[1]
        event_type = desc_tokens[1]
        gcode = desc_tokens[2]
        uuid = desc_tokens[3]
        proj = desc_tokens[4]

        # if the event description starts with 8 characters followed by a . and numbers, this is a subtask uuid.
        subtask_uuid = ''
        event_desc = ''
        is_subtask = '.' in desc_tokens[5] and len(desc_tokens[5]) >= len('badfeed8.01') and desc_tokens[5][8] == '.'
        has_subtask = '.' not in desc_tokens[5] and len(desc_tokens[5]) == len('badfeed8') and \
                      all(desc_tokens[5] | pipe.map(lambda c: c >= '0' and c <= 'f'))
        if is_subtask or has_subtask:
            subtask_uuid = desc_tokens[5]
            event_desc = ' '.join(desc_tokens[6:])
        else:
            event_desc = ' '.join(desc_tokens[5:])

        # Parse the begin time, should just be from the begin_time_s and the csv year month day.
        strptime_datetime_format = '%Y-%m-%d %H:%M'
        begin_dt = dtdt.strptime(f'{_year}-{_month}-{_day} {begin_time_s_hh}:{begin_time_s_mm}', strptime_datetime_format)
        interval = timedelta(hours=int(interval_time_s[:2]), minutes=int(interval_time_s[2:]))
        end_dt = begin_dt + interval




        return CalcureEvent(_id, begin_dt, interval, end_dt, event_type, gcode, uuid, subtask_uuid, proj, event_desc, int(_repeat_count), _repeat_state, _priority)

    def to_csv(self) -> str:
        # ['217', '2023', '7', '7', '1340:0021 EVNT DRV1 afd865ee drvlic Request Driving car on exam time', '1', 'once', 'unimportant']
        subtask_uuid = f'{self.subtask_uuid} ' if len(self.subtask_uuid) != 0 else ''
        desc = f'{self.begin_dt.hour:02d}{self.begin_dt.minute:02d}:{self.interval.seconds // 3600:02d}{(self.interval.seconds // 60) % 60:02d} {self.event_type} {self.gcode} {self.uuid} {self.proj} {subtask_uuid}{self.desc}'
        return f'{self._id},{self.begin_dt.year},{self.begin_dt.month},{self.begin_dt.day},\"{desc}\",{str(self.repeat_count)},{self.repeat_state},{self.priority}'


class TimeSlots:
    # Represent time slots on a 5-minute basis, 288 5-minutes in the day.
    def __init__(self):
        self.slots = [None] * 288  # Fill slots with None, representing no event

    @staticmethod
    def index(t: timedelta) -> int:
        """
        Given a time, gives the index of it in the timeslots. This would just be which multiple of 5 minutes it is.
        """
        return t.seconds // 60 // 5

    @staticmethod
    def time(i: index) -> timedelta:
        return timedelta(minutes=i * 5)

    def event(self, t: timedelta) -> Optional[Tuple[int, timedelta, timedelta]]:
        """
        Given a time, give the event_id, start time, and duration of an event allocated there.
        """
        idx = self.index(t)
        event_id = self.slots[idx]

        if event_id is not None:
            # Find the start of the event
            start_idx = idx
            while start_idx > 0 and self.slots[start_idx - 1] == event_id:
                start_idx -= 1

            # Find the end of the event
            end_idx = idx
            while end_idx < len(self.slots) - 1 and self.slots[end_idx + 1] == event_id:
                end_idx += 1

            # Compute start time and duration
            start_time = timedelta(minutes=start_idx * 5)
            duration = timedelta(minutes=(end_idx - start_idx + 1) * 5)

            return event_id, start_time, duration

        else:
            return None


    def add_event(self, event_id: int, t: timedelta, duration: timedelta):
        """
        Adds an event at a given time for a certain duration.
        Note that conflicts are allowed and this only overrides.
        """
        start_idx = self.index(t)
        end_idx = self.index(t + duration)
        for idx in range(start_idx, end_idx):
            self.slots[idx] = event_id

    def event_fits(self, t: timedelta, duration: timedelta) -> bool:
        """
        Checks whether an event can be added.
        """
        start_idx = self.index(t)
        end_idx = self.index(t + duration)
        if end_idx < start_idx:
            return False
        return all(slot is None for slot in self.slots[start_idx:end_idx])

    def display(self):
        for i, slot in enumerate(self.slots):
            print(f'{i * 5 // 60:02d}:{(i * 5) % 60:02d} {i} {slot}')


class TaskEventReporting:
    @staticmethod
    def visualize_treemap_tasks(args):
        import plotly.express as px
        import textwrap

        cls = TaskEventReporting

        events = cls.read_calcure_events()
        df_tasks = cls.read_tasks_dataframe()
        num_subtask_layers = 5

        # list(events | pipe.map(lambda e: e.to_csv()) | pipe.map(lambda e: print(e)))

        # Let's create a dataframe with maximum N levels of task depth. The treemap path would
        # look like: gcode - project - supertask - [(subtask or pd.NA) xN] - event - event duration - path


        if args.filter_week:
            filter_for_week = cls.parse_weekdate(args.filter_week)
            events = list(events | pipe.filter(lambda e: e.begin_dt >= filter_for_week and
                                                         e.begin_dt < filter_for_week + timedelta(days=args.mult_week * 7)))

        if args.filter_day:
            filter_for_day = cls.parse_weekdate(args.filter_day)
            events = list(events | pipe.filter(lambda e: e.begin_dt >= filter_for_day and
                                                         e.begin_dt < filter_for_day + timedelta(days=args.mult_day)))

        if args.filter_uuid:
            events = list(events | pipe.filter(lambda e: e.uuid == args.filter_uuid or 
                                               args.filter_uuid in e.subtask_uuid))

        if args.filter_pattern:
            events = list(events | pipe.filter(lambda e: args.filter_pattern in e.desc))

        cols_viz = ['gcode', 'project', 'supertask', 'event']
        for i in range(num_subtask_layers):
            cols_viz.append(f'subtask{i}')
        cols_viz.append('dur')
        cols_viz.append('path')
        cols_viz.append('week')
        cols_viz.append('hour')
        if args.cluster_week:
            cols_viz.append('cur_week')
        if args.cluster_day:
            cols_viz.append('cur_day')

        df_viz = pd.DataFrame(columns=cols_viz)

        # go through all events, and figure out task structure along each event, keeping that event
        # references last
        for event in events:
            def task_to_str(task: pd.Series) -> str:
                desc = task["description"]

                # remove any uuid
                desc_tokens = desc.split(' ')
                if len(desc_tokens[0]) == len('899e9e05') or '.' in desc_tokens[0]:
                    desc = ' '.join(desc_tokens[1:])

                return desc

            def event_to_str(event: CalcureEvent) -> str:
                desc = event.desc

                if args.no_event_timestamps:
                    # remove any stats at the start of the event
                    event_desc_tokens = event.desc.split(' ')
                    if ')' in event_desc_tokens[0]:
                        desc = ' '.join(event_desc_tokens[1:])
                else:
                    cur_day = cls.convert_to_short_daycode(''.join(reversed(''.join(reversed(event.begin_dt.strftime('%G%m%d-W%V%a')))[:-2])))
                    cur_day = cur_day.split('-')[1]
                    timestamp = ' '.join(event.to_csv().split(' ')[:1]).replace('"', '')
                    timestamp = ','.join(timestamp.split(',')[1:])

                    timestamp_date = '/'.join(timestamp.split(',')[:-1])
                    timestamp_interval = timestamp.split(',')[-1]

                    # desc = cur_day + ' ' + timestamp + ' ' + desc
                    desc = timestamp_interval + ' ' + desc + '<br>' + timestamp_date + ' ' + cur_day

                return desc

            def get_task_with_text(s):
                # FIXME should return multiple
                mask = df_tasks['description'].str.contains(s)
                return df_tasks[mask].iloc[0]

            def get_task_with_uuid(uuid):
                try:
                    mask = df_tasks['uuid'].str.contains(uuid)
                    return df_tasks[mask].iloc[0]
                except Exception:
                    print(f'error finding tssk with uuid {uuid}')
                    raise


            if event.priority != args.priority:
                continue

            if args.skip_gcodes:
                if event.gcode in args.skip_gcodes:
                    continue
            
            if args.filter_gcodes:
                if event.gcode not in args.filter_gcodes:
                    continue

            if event.subtask_uuid != '':
                subtask_uuid_tokens = event.subtask_uuid.split('.')
                supertask_uuid = subtask_uuid_tokens[0]
                subtask_counts = []
                if len(subtask_uuid_tokens) > 1:
                    subtask_counts = subtask_uuid_tokens[1:]

                event_tasks = [get_task_with_text(f'{supertask_uuid} ')]

                uuid = f'{supertask_uuid}'
                for count in subtask_counts:
                    uuid = f'{uuid}.{count}'
                    event_tasks.append(get_task_with_text(f'{uuid} '))

                dict_viz = {'gcode': event.gcode, 'project': event.proj, 'supertask': task_to_str(event_tasks[0]),
                            'event': event_to_str(event), 'dur': event.interval.seconds / 3600, 
                            'path': ['gcode', 'project', 'supertask']}

                if len(event_tasks) > 1:
                    for i, event_task in enumerate(event_tasks[1:]):
                        subtask_col = f'subtask{i}'
                        dict_viz[subtask_col] = task_to_str(event_tasks[i+1])
                        dict_viz['path'].append(subtask_col)

                    for i in range(len(event_tasks[1:]), num_subtask_layers):
                        dict_viz[f'subtask{i}'] = " "
                else:
                    for i in range(num_subtask_layers):
                        dict_viz[f'subtask{i}'] = " "

            else:
                task = get_task_with_uuid(event.uuid)
                dict_viz = {'gcode': event.gcode, 'project': event.proj, 'supertask': task_to_str(task),
                            'event': event_to_str(event), 'dur': event.interval.seconds / 3600, 
                            'path': ['gcode', 'project', 'supertask']}

                for i in range(num_subtask_layers):
                    dict_viz[f'subtask{i}'] = " "

            if args.cluster_week:
                dict_viz['cur_week'] = ''.join(reversed(''.join(reversed(event.begin_dt.strftime('%G-W%V')))[:-2]))
            if args.cluster_day:
                dict_viz['cur_day'] = cls.convert_to_short_daycode(''.join(reversed(''.join(reversed(event.begin_dt.strftime('%G%m%d-W%V%a')))[:-2])))
            dict_viz['week'] =  (((event.begin_dt - dtdt(year=2023, month=1, day=1)).days / 7) + 1)
            dict_viz['week'] = int(dict_viz['week'] * 1000) / 1000
            dict_viz['hour'] =  float(event.begin_dt.hour)


            df_viz = pd.concat([df_viz, pd.DataFrame([dict_viz])], ignore_index=True)



        path_cols = [col for col in cols_viz if df_viz[col].notna().any()]
        print(path_cols)

        df_viz['color'] = 'blue'
        path = ['gcode', 'project', 'supertask', 'subtask0', 'event']

        if args.cluster_day:
            path = ['cur_day'] + path
        if args.cluster_week:
            path = ['cur_week'] + path

        title = f'Treemap of time spent on events and their task structure'

        title += ' ('
        title += f'priority: {args.priority.replace("unimportant", "done")}, '

        if args.filter_day:
            if args.mult_day != 1:
                title += f'filter_for_days + {args.mult_day} days: {args.filter_day}, '
            else:
                title += f'filter_for_day: {args.filter_day}, '
        if args.filter_week:
            if args.mult_week != 1:
                title += f'filter_for_weeks + {args.mult_week} weeks: {args.filter_week[0:4] + "-" + args.filter_week.split("-")[1]}, '
            else:
                title += f'filter_for_week: {args.filter_week[0:4] + "-" + args.filter_week.split("-")[1]}, '
        if args.filter_uuid:
                title += f'filter_by_uuid: {args.filter_uuid}, '
        if args.filter_pattern:
                title += f'filter_by_pattern: {args.filter_pattern}, '
        if args.no_event_timestamps:
                title += f'no_event_timestamps, '
        title += ')'
        title = textwrap.fill(title, width=120)
        title = title.replace('\n', '<br>')
        print(title)

        # Skip 0-sized values. They can't be normalized.
        df_viz = df_viz[df_viz['dur'] != 0]

        print(df_viz)
        df_viz.to_csv('here.csv')

        fig = px.treemap(df_viz, path=path, 
                         values='dur', color=args.treemap_color, color_discrete_sequence=['blue'],
                         title=title)

        fig.update_traces(hovertemplate='%{label}<br>%{value} hours<extra></extra>')

        fig.show()

    @staticmethod
    def read_calcure_events() -> List[CalcureEvent]:
        CALCURE_EVENTS_CSV_PATH = 'events.csv'
        result = []

        with open(CALCURE_EVENTS_CSV_PATH, 'r') as calcure_events_file:
            csv_reader = csv.reader(calcure_events_file)
            for row in csv_reader:
                result.append(CalcureEvent.read_csv(row))
        
        return result

    @staticmethod
    def read_tasks_dataframe() -> pd.DataFrame:
        import random
        rand = random.random()

        # os.system('task export > {rand}.json')
        # df = pd.read_json(f'/tmp/{rand}.json')

        df = pd.read_json('tasks.json')

        # print(df)
        # print(df.shape)

        # os.remove('/tmp/{rand}.json')

        return df

    def convert_to_long_daycode(s: str) -> str:
        if s[-1] == 'M':
            s = s[:-1] + 'Mon'
        elif s[-1] == 'T':
            s = s[:-1] + 'Tue'
        elif s[-1] == 'W':
            s = s[:-1] + 'Wed'
        elif s[-1] == 'R':
            s = s[:-1] + 'Thu'
        elif s[-1] == 'F':
            s = s[:-1] + 'Fri'
        elif s[-1] == 'S':
            s = s[:-1] + 'Sat'
        elif s[-1] == 'U':
            s = s[:-1] + 'Sun'
        
        return s

    def convert_to_short_daycode(s: str) -> str:
        if s.endswith('Mon'):
            s = s[:-3] + 'M'
        elif s.endswith('Tue'):
            s = s[:-3] + 'T'
        elif s.endswith('Wed'):
            s = s[:-3] + 'W'
        elif s.endswith('Wed'):
            s = s[:-3] + 'W'
        elif s.endswith('Thu'):
            s = s[:-3] + 'R'
        elif s.endswith('Fri'):
            s = s[:-3] + 'F'
        elif s.endswith('Sat'):
            s = s[:-3] + 'S'
        elif s.endswith('Sun'):
            s = s[:-3] + 'U'
        
        return s

    @staticmethod
    def parse_weekdate(s: str) -> dtdt:
        cls = TaskEventReporting
        # 230828-W35M
        if s[-1].isnumeric():
            s = s + 'Mon'
        else:
            s = cls.convert_to_long_daycode(s)
            
        return dtdt.strptime('20' + s, '%G%m%d-W%V%a')

def ConvertCalcureEventsToMarkersJson():
    result = {}
    result['markers'] = []

    with open(CALCURE_EVENTS_CSV_PATH, 'r') as calcure_events_file:
        csv_reader = csv.reader(calcure_events_file)
        for row in csv_reader:
            calcure_event = CalcureEvent.read_csv(row)

            # figure out background based on priority
            background = ''
            if calcure_event.priority == 'unimportant':
                background = 'rgba(97,214,214,1)'
            if calcure_event.priority == 'important':
                background = 'rgba(249,241,165,1)'

            subtask_uuid_desc = f'subtask_uuid: {calcure_event.subtask_uuid}, ' if len(calcure_event.subtask_uuid) != 0 else ''

            schedule_event = {
                "id": calcure_event._id,
                #"title": f"{calcure_event.event_type} {calcure_event.gcode} {calcure_event.uuid} {calcure_event.proj} {calcure_event.desc}",
                "title": f"{calcure_event.gcode} {calcure_event.proj} {calcure_event.desc}",
                "begin": calcure_event.begin_dt.strftime('%Y/%m/%d %H:%M:%S'),
                "end": calcure_event.end_dt.strftime('%Y/%m/%d %H:%M:%S'),
                "description": f"event: {calcure_event.event_type}, uuid: {calcure_event.uuid}, {subtask_uuid_desc} interval: {calcure_event.interval.seconds // 3600:02d}:{(calcure_event.interval.seconds // 60) % 60:02d}",
                "background": background,
            }
            result['markers'].append(schedule_event)
            pass
    
    with open(MARKERS_JSON_PATH, 'w') as json_file:
        json_file.write(json.dumps(result))
    time.sleep(8)

    # Due to React update bug, double-trigger changes so that the screen updates to the latest. This will be shortly reverted.
    #  with open(MARKERS_JSON_PATH, 'w') as json_file:
        #  json_file.write(json.dumps({"markers": [{"id":1688965338515,"title":"AAA","begin":"2023/07/09 03:30:00","end":"2023/07/09 04:30:00","description":"", "background": "rgba(256,0,0,1)"}]}))z

    with open(MARKERS_JSON_PATH, 'w') as json_file:
        json_file.write(json.dumps(result))


    
# 1688965338515
# {id:1688965338515,title:AAA,begin:2023/07/09 03:30:00,end:2023/07/09 04:30:00,description:"", background: rgba(256,0,0,1)},
    pass

def WatchCalcureEventsAndUpdateMarkersJson():
    def check_hash():
        with open(CALCURE_EVENTS_CSV_PATH, 'rb') as file:
            return hashlib.sha256(file.read()).hexdigest()

    stored_hash = check_hash()

    print(f'{dtdt.now()} Watching for changes in calcure events to update public/markers.json.\n\t{stored_hash[:6]}')

    while True:
        try:
            new_hash = check_hash()
            if new_hash != stored_hash:
                print(f'{dtdt.now()} Detected calcure event change. Updating public/markers.json.\n\t{stored_hash[:6]} -> {new_hash[:6]}')
                ConvertCalcureEventsToMarkersJson()
                stored_hash = new_hash
        except FileNotFoundError:
            print(f'{dtdt.now()} Calcure events file deleted.')
        except Exception as e:
            print(f'{dtdt.now()} Calcure Watch Exception: {e}')

        time.sleep(1)

def ExpandCalcureEventsToOrderedSequencialEvents(begin: timedelta, first_id, last_id):
    with open(CALCURE_EVENTS_CSV_PATH, 'r') as calcure_events_file:
        csv_reader = csv.reader(calcure_events_file)
        for row in csv_reader:
            calcure_event = CalcureEvent.read_csv(row)
            #  same_day = calcure_event.begin_dt.year == day.year and calcure_event.begin_dt.month == day.month and calcure_event.begin_dt.day == day.day
            within_id_range = int(calcure_event._id) >= first_id and int(calcure_event._id) < last_id
            if not within_id_range:
                continue
            calcure_event.begin_dt += begin
            begin += calcure_event.interval + timedelta(minutes=10)
            print(calcure_event.to_csv())

def SortPriorityFloatingStartTimeEvents(day: dtdt, begin: timedelta, skip_missed: bool):
    def PopulateCurrentFixedEvents(timeslots: TimeSlots, day: dtdt, skip_missed: bool) -> dict:
        result = {
            'floating_events': [],
            'fixed_events': [],
        }

        with open(CALCURE_EVENTS_CSV_PATH, 'r') as calcure_events_file:
            csv_reader = csv.reader(calcure_events_file)
            for row in csv_reader:
                calcure_event = CalcureEvent.read_csv(row)

                # Make sure that we're not packing ourselves too much with 10 minute paddding
                event_begin = timedelta(hours=calcure_event.begin_dt.hour, minutes=calcure_event.begin_dt.minute)
                event_dur = calcure_event.interval + timedelta(minutes=10)
                event_dur_minutes = event_dur.seconds // 60

                # we're only concerned with mapping today
                same_day = calcure_event.begin_dt.year == day.year and calcure_event.begin_dt.month == day.month \
                           and calcure_event.begin_dt.day == day.day
                if not same_day:
                    continue

                # check if the event is completed/ongoing or if it has an interval ending with 1 or 6 (non-floating signal)
                fixed = calcure_event.priority == 'unimportant' or calcure_event.priority == 'important' or \
                        event_dur_minutes % 10 == 1 or event_dur_minutes % 10 == 6

                # Anything with a beginning before specified beginning will be skipped.
                if skip_missed:
                    fixed = fixed or event_begin.seconds != 0 and event_begin < begin

                # Some are floating but we should not touch them. 00:01-00:04
                ignored_floating = event_begin.seconds // 60 > 0 and event_begin.seconds // 60 < 5

                if fixed or ignored_floating:
                    timeslots.add_event(calcure_event._id, event_begin, event_dur)
                    result['fixed_events'].append(calcure_event)
                else:
                    result['floating_events'].append(calcure_event)
        return result

    # if begin is 00:00, use current time
    if (begin.seconds == 0):
        begin = timedelta(hours=dtdt.now().hour, minutes=dtdt.now().minute)

    #today = dtdt.now() - timedelta(hours=dtdt.now().hour, minutes=dtdt.now().minute, seconds=dtdt.now().second)

    timeslots = TimeSlots()
    events_dict = PopulateCurrentFixedEvents(timeslots, day, skip_missed)

    #timeslots.display()

    for calcure_event in events_dict['fixed_events']:
        print(calcure_event.to_csv())
    print('')

    for calcure_event in events_dict['floating_events']:
        event_begin = timedelta(hours=calcure_event.begin_dt.hour, minutes=calcure_event.begin_dt.minute)
        event_dur = calcure_event.interval + timedelta(minutes=10) # +10 minutes for padding
        event_dur_minutes = event_dur.seconds // 60

        # Find where the event can fit next
        begin_slot = timeslots.index(begin)
        while begin_slot < len(timeslots.slots) - 1 and not timeslots.event_fits(timeslots.time(begin_slot), event_dur):
            begin_slot += 1

        calcure_event.begin_dt = day + timeslots.time(begin_slot)
        begin = timeslots.time(begin_slot) + event_dur

        #print(f'bs={begin_slot}, t={timeslots.time(begin_slot)}, begin={begin}, begin_seconds={begin.seconds}, >day?{begin.seconds >= 60*60*24}, dur={event_dur}, fits?{timeslots.event_fits(timeslots.time(begin_slot), event_dur)}')

        if begin_slot >= len(timeslots.slots) - 1 or timeslots.index(begin) >= len(timeslots.slots) - 1 or \
           begin.seconds == 0 and begin.days > 0:
            print(f'error: Could not fit event into day\n\t{calcure_event.to_csv()}')
            continue

        #print(begin_slot, calcure_event.to_csv())
        print(calcure_event.to_csv())

def OpTimesAndDisplay(t1: timedelta, op: str, t2: timedelta):
    if op == '-':
        result = t1 - t2
    elif op == '+':
        result = t1 + t2
    else:
        raise Exception(f'unknown op {op}')

    return f'{result.seconds // 3600:02d}:{(result.seconds // 60) % 60:02d}'


def main(args):
    if args.subcommand == 'populate':
        ConvertCalcureEventsToMarkersJson()
    if args.subcommand == 'watch':
        WatchCalcureEventsAndUpdateMarkersJson()
    if args.subcommand == 'old_order':
        hours, minutes = map(int, args.hhmm.split(':'))
        i, j = map(int, args.ij.split(':'))
        ExpandCalcureEventsToOrderedSequencialEvents(timedelta(hours=hours, minutes=minutes), i, j)
    if args.subcommand == 'sort_float':
        day = dtdt.strptime(args.day, '%Y-%m-%d')
        hours, minutes = map(int, args.hhmm.split(':'))
        skip_missed = args.skip_missed
        SortPriorityFloatingStartTimeEvents(day, timedelta(hours=hours, minutes=minutes), skip_missed)
    if args.subcommand == 'op':
        hours1, minutes1 = map(int, args.hhmm1.split(':'))
        op = args.op
        hours2, minutes2 = map(int, args.hhmm2.split(':'))
        print(OpTimesAndDisplay(timedelta(hours=hours1, minutes=minutes1), op,
                                timedelta(hours=hours2, minutes=minutes2)))
    if args.subcommand == 'visualize_tasks':
        TaskEventReporting.visualize_treemap_tasks(args)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Commands to control Calcure Schedule Gui and Visualizations")
    subparsers = parser.add_subparsers(dest="subcommand", required=True, help="Subcommands")

    subparser = subparsers.add_parser("populate", 
                                      help="Subcommand 1 description")

    subparser = subparsers.add_parser("watch", 
                                      help="Subcommand 1 description")

    # subparser = subparsers.add_parser("old_order", 
    #                                   help="Subcommand 1 description")
    # subparser.add_argument("hhmm", type=str, help="HH:MM")
    # subparser.add_argument("ij", type=str, help="%d:%d")
    # subparser.add_argument("--skip_missed", type=bool, help="Skip times that have already been accounted for")

    subparser = subparsers.add_parser("sort_float", 
                                      help="Subcommand 1 description")
    subparser.add_argument("day", type=str, help="Format of %Y-%m-%d")
    subparser.add_argument("hhmm", type=str, help="HH:MM")
    subparser.add_argument("--skip_missed", type=bool, help="Skip times that have already been accounted for")

    subparser = subparsers.add_parser("op", 
                                      help="Subcommand 1 description")
    subparser.add_argument("hhmm1", type=str, help="HH:MM")
    subparser.add_argument("op", type=str, help="operation")
    subparser.add_argument("hhmm2", type=str, help="HH:MM")

    subparser = subparsers.add_parser("visualize_tasks", 
                                      help="Subcommand 1 description")
    subparser.add_argument("--no-event-timestamps", action="store_true", help="Don't Include event timestamps")
    subparser.add_argument("--filter-week", type=str, help="Week to filter for")
    subparser.add_argument("--mult-week", type=int, default=1, help="Multiplier for week filter")
    subparser.add_argument("--filter-day", type=str, help="Day to filter for. 23MMDD-WVV[MTWRSU]")
    subparser.add_argument("--mult-day", type=int, default=1, help="Multiplier for day filter")
    subparser.add_argument("--filter-uuid", type=str, help="UUID to filter by")
    subparser.add_argument("--filter-pattern", type=str, help="Pattern to filter by")
    subparser.add_argument("--skip-gcodes", type=str, nargs='+', help="Gcodes to skip")
    subparser.add_argument("--filter-gcodes", type=str, nargs='+', help="Gcodes to filter by")
    subparser.add_argument("--cluster-week", action="store_true", help="Cluster by week")
    subparser.add_argument("--cluster-day", action="store_true", help="Cluster by day")
    subparser.add_argument("--priority", type=str, choices=['unimportant', 'normal'], default='unimportant', help="Set priority")
    subparser.add_argument("--treemap-color", type=str, choices=['week', 'hour'], default='week', help="Set treemap color")

    # Parse the arguments
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main(parse_args())