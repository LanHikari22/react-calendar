import sys
import csv
import json
import hashlib
import time
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
        if '.' in desc_tokens[5] and len(desc_tokens[5]) >= len('badfeed8.01') and desc_tokens[5][8] == '.':
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

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print ('usage: calcure_events {populate|watch|old_order|sort_float|op}')
        exit(0)

    if sys.argv[1] == 'populate':
        ConvertCalcureEventsToMarkersJson()
    if sys.argv[1] == 'watch':
        WatchCalcureEventsAndUpdateMarkersJson()
    if sys.argv[1] == 'old_order':
        if len(sys.argv) != 4:
            print('usage: calcure_event sort_order {HH:MM} {i:j}')
            exit(0)
        hours, minutes = map(int, sys.argv[2].split(':'))
        i, j = map(int, sys.argv[3].split(':'))
        ExpandCalcureEventsToOrderedSequencialEvents(timedelta(hours=hours, minutes=minutes), i, j)
    if sys.argv[1] == 'sort_float':
        if len(sys.argv) != 5:
            print('usage: calcure_event sort_float {YYYY-MM-DD} {HH:MM|00:00} {[no_]skip_missed}')
            exit(0)
        day = dtdt.strptime(sys.argv[2], '%Y-%m-%d')
        hours, minutes = map(int, sys.argv[3].split(':'))
        skip_missed = sys.argv[4] == 'skip_missed'
        SortPriorityFloatingStartTimeEvents(day, timedelta(hours=hours, minutes=minutes), skip_missed)
    if sys.argv[1] == 'op':
        if len(sys.argv) != 5:
            print('usage: calcure_event op {HH:MM} {+|-} {HH:MM}')
            exit(0)
        hours1, minutes1 = map(int, sys.argv[2].split(':'))
        op = sys.argv[3]
        hours2, minutes2 = map(int, sys.argv[4].split(':'))
        print(OpTimesAndDisplay(timedelta(hours=hours1, minutes=minutes1), op,
                                timedelta(hours=hours2, minutes=minutes2)))
