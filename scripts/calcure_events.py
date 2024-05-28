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
    class TreemapTasksVisualizer:
        def __init__(self, args):
            self.args = args

        def event_to_str(self, event: CalcureEvent) -> str:
            desc = event.desc
            parent_cls = TaskEventReporting

            if self.args.no_event_timestamps:
                # remove any stats at the start of the event
                event_desc_tokens = event.desc.split(' ')
                if ')' in event_desc_tokens[0]:
                    desc = ' '.join(event_desc_tokens[1:])
            else:
                cur_day = parent_cls.convert_to_short_daycode(''.join(reversed(''.join(reversed(event.begin_dt.strftime('%G%m%d-W%V%a')))[:-2])))
                cur_day = cur_day.split('-')[1]
                timestamp = ' '.join(event.to_csv().split(' ')[:1]).replace('"', '')
                timestamp = ','.join(timestamp.split(',')[1:])

                timestamp_date = '/'.join(timestamp.split(',')[:-1])
                timestamp_interval = timestamp.split(',')[-1]

                # desc = cur_day + ' ' + timestamp + ' ' + desc
                desc = timestamp_interval + ' ' + desc + '<br>' + timestamp_date + ' ' + cur_day

            return desc

        def visualize(self):
            import plotly.express as px
            import textwrap

            cls = TaskEventReporting.TreemapTasksVisualizer
            parent_cls = TaskEventReporting

            events = parent_cls.read_calcure_events()
            df_tasks = parent_cls.read_tasks_dataframe()

            using_variable_subtasks = True
            num_subtask_layers = 4

            # list(events | pipe.map(lambda e: e.to_csv()) | pipe.map(lambda e: print(e)))

            # Let's create a dataframe with maximum N levels of task depth. The treemap path would
            # look like: gcode - project - supertask - [(subtask or pd.NA) xN] - event - event duration - path

            if self.args.filter_week:
                filter_for_week = parent_cls.parse_weekdate(self.args.filter_week)
                events = list(events | pipe.filter(lambda e: e.begin_dt >= filter_for_week and
                                                            e.begin_dt < filter_for_week + timedelta(days=self.args.mult_week * 7)))

            if self.args.filter_day:
                filter_for_day = parent_cls.parse_weekdate(self.args.filter_day)
                events = list(events | pipe.filter(lambda e: e.begin_dt >= filter_for_day and
                                                            e.begin_dt < filter_for_day + timedelta(days=self.args.mult_day)))

            if self.args.filter_uuid:
                events = list(events | pipe.filter(lambda e: e.uuid == self.args.filter_uuid or 
                                                self.args.filter_uuid in e.subtask_uuid))

            if self.args.filter_pattern:
                events = list(events | pipe.filter(lambda e: self.args.filter_pattern in e.desc))

            cols_viz = ['gcode', 'project', 'supertask']
            if not using_variable_subtasks:
                for i in range(num_subtask_layers):
                    cols_viz.append(f'subtask{i}')
            cols_viz.append('event')
            cols_viz.append('dur')
            cols_viz.append('path')
            cols_viz.append('week')
            cols_viz.append('hour')
            if self.args.cluster_week:
                cols_viz.append('cur_week')
            if self.args.cluster_day:
                cols_viz.append('cur_day')

            df_viz = pd.DataFrame(columns=cols_viz)

            # go through all events, and figure out task structure along each event, keeping that event
            # references last
            for event in events:
                if event.priority != self.args.priority:
                    continue

                if self.args.skip_gcodes:
                    if event.gcode in self.args.skip_gcodes:
                        continue
                
                if self.args.filter_gcodes:
                    if event.gcode not in self.args.filter_gcodes:
                        continue

                if event.subtask_uuid != '': # We're some subtask
                    # This event task has parents we need to account for
                    event_task_branch = [cls.get_task_with_uuid(df_tasks, event.uuid)]

                    # Get our parent tasks
                    parent_uuids = cls.get_parents_of_uuid(df_tasks, event.uuid)

                    # Account for each parent leading up to root
                    for uuid in parent_uuids:
                        # print(f'uuid: \"{uuid}\"')
                        event_task_branch.append(cls.get_task_with_uuid(df_tasks, uuid))
                    
                    if not using_variable_subtasks:
                        # Ensure to trim the branch to num_subtask_layers, as we have a set limit.
                        if len(event_task_branch) > num_subtask_layers:
                            event_task_branch = event_task_branch[:num_subtask_layers]

                    # Reverse the branch to start from root, our supertask as it is called.
                    event_task_branch = list(reversed(event_task_branch))

                    dict_viz = {'gcode': event.gcode, 'project': event.proj, 
                                'supertask': cls.task_to_str(event_task_branch[0]),
                                'event': self.event_to_str(event), 'dur': event.interval.seconds / 3600, 
                                'path': ['gcode', 'project', 'supertask']}
                    
                    if len(event_task_branch) > 1: # We were able to find parents

                        # Fill in the the subtasks in the branch. We counted root.
                        for i, subtask in enumerate(event_task_branch[1:]):
                            subtask_col = f'subtask{i}'
                            dict_viz[subtask_col] = cls.task_to_str(event_task_branch[i+1])
                            dict_viz['path'].append(subtask_col)

                        if not using_variable_subtasks:
                            # If any are remaining from our `num_subtask_layers` limit, fill with empties.
                            for i in range(len(event_task_branch[1:]), num_subtask_layers):
                                dict_viz[f'subtask{i}'] = " "
                    else: # For some reason, we expected parents but could not find any.
                        # print(f'Warning: Task {event.uuid} expected to have parents but it does not.')

                        if not using_variable_subtasks:
                            # Fill in all subtask columns with empties.
                            for i in range(num_subtask_layers):
                                dict_viz[f'subtask{i}'] = " "
                else: # We're a root element, no subtasks.
                    task = cls.get_task_with_uuid(df_tasks, event.uuid)
                    dict_viz = {'gcode': event.gcode, 'project': event.proj, 
                                'supertask': cls.task_to_str(task),
                                'event': self.event_to_str(event), 'dur': event.interval.seconds / 3600, 
                                'path': ['gcode', 'project', 'supertask']}

                    if not using_variable_subtasks:
                        # Fill in all subtask columns with empties.
                        for i in range(num_subtask_layers):
                            dict_viz[f'subtask{i}'] = " "

                if self.args.cluster_week:
                    dict_viz['cur_week'] = ''.join(reversed(''.join(reversed(event.begin_dt.strftime('%G-W%V')))[:-2]))
                if self.args.cluster_day:
                    dict_viz['cur_day'] = parent_cls.convert_to_short_daycode(''.join(reversed(''.join(reversed(event.begin_dt.strftime('%G%m%d-W%V%a')))[:-2])))

                dict_viz['week'] =  (((event.begin_dt - dtdt(year=2023, month=1, day=1)).days / 7) + 1)
                dict_viz['week'] = int(dict_viz['week'] * 1000) / 1000
                dict_viz['hour'] =  float(event.begin_dt.hour)

                df_viz = pd.concat([df_viz, pd.DataFrame([dict_viz])], ignore_index=True)


            path_cols = [col for col in cols_viz if df_viz[col].notna().any()]
            print(path_cols)

            df_viz['color'] = 'blue'

            # path = ['gcode', 'project', 'supertask', 'subtask0', 'event']
            path = ['gcode', 'project', 'supertask']
            for i in range(num_subtask_layers):
                path.append(f'subtask{i}')
            path.append('event')

            if self.args.cluster_day:
                path = ['cur_day'] + path
            if self.args.cluster_week:
                path = ['cur_week'] + path

            title = f'Treemap of time spent on events and their task structure'

            title += ' ('
            title += f'priority: {self.args.priority.replace("unimportant", "done")}, '

            if self.args.filter_day:
                if self.args.mult_day != 1:
                    title += f'filter_for_days + {self.args.mult_day} days: {self.args.filter_day}, '
                else:
                    title += f'filter_for_day: {self.args.filter_day}, '
            if self.args.filter_week:
                if self.args.mult_week != 1:
                    title += f'filter_for_weeks + {self.args.mult_week} weeks: {self.args.filter_week[0:4] + "-" + self.args.filter_week.split("-")[1]}, '
                else:
                    title += f'filter_for_week: {self.args.filter_week[0:4] + "-" + self.args.filter_week.split("-")[1]}, '
            if self.args.filter_uuid:
                    title += f'filter_by_uuid: {self.args.filter_uuid}, '
            if self.args.filter_pattern:
                    title += f'filter_by_pattern: {self.args.filter_pattern}, '
            if self.args.no_event_timestamps:
                    title += f'no_event_timestamps, '
            title += ')'
            title = textwrap.fill(title, width=120)
            title = title.replace('\n', '<br>')
            print(title)

            # Skip 0-sized values. They can't be normalized.
            df_viz = df_viz[df_viz['dur'] != 0]

            if using_variable_subtasks:
                # Let's reorder to make sure that subtaskN follow after supertask
                df_viz = cls.reorder_variable_subtasks(df_viz)


            print(df_viz)
            df_viz.to_csv('here.csv')

            if self.args.text_only:
                df_text_only = self.compute_accumulated_duration(df_viz)
                df_text_only.to_csv('df_text_only.csv')
                cls.display_accumulated_duration_df(df_text_only)

            # print('path', path)

            if not self.args.text_only:
                fig = px.treemap(df_viz, path=path, 
                                values='dur', color=self.args.treemap_color, color_discrete_sequence=['blue'],
                                title=title)

                fig.update_traces(hovertemplate='%{label}<br>%{value} hours<extra></extra>')

                fig.show()

        @staticmethod
        def task_to_str(task: pd.Series) -> str:
            desc = task["description"]

            # remove any uuid
            desc_tokens = desc.split(' ')
            if len(desc_tokens[0]) == len('899e9e05') or '.' in desc_tokens[0]:
                desc = ' '.join(desc_tokens[1:])

            return desc

        @staticmethod
        def get_task_with_text(df_tasks, s):
            # FIXME should return multiple
            try:
                mask = df_tasks['description'].str.contains(s)
                return df_tasks[mask].iloc[0]
            except IndexError as e:
                print(f'Could not find any task with text "{s}"')
                raise

        @staticmethod
        def get_task_with_uuid(df_tasks, uuid):
            try:
                mask = df_tasks['uuid'].str.contains(uuid)
                return df_tasks[mask].iloc[0]
            except Exception:
                    print(f'error finding task with uuid \"{uuid}\"')
                    raise

        @staticmethod
        def get_parents_of_uuid(df_tasks, uuid):
            try:
                cls = TaskEventReporting.TreemapTasksVisualizer
                result = []
                task = cls.get_task_with_uuid(df_tasks, uuid)

                childof = task['childof']
                if type(childof) != str:
                    # print(f'Warning: task {uuid} does not have str childof but instead uses ' + 
                    #       f'{type(childof)}: {childof}')
                    return []

                tokens = list(reversed(list(childof.split(','))))
                # print('childof', task['childof'])
                # print('tokens', tokens)

                for token in tokens:
                    if token.strip() == '':
                        continue
                    result.append(token)
                return result
            except Exception:
                print(f'error finding parents of uuid \"{uuid}\"')
                raise

        @staticmethod
        def reorder_variable_subtasks(df):
            # Identify the position of the 'supertask' column
            supertask_index = df.columns.get_loc('supertask')

            # Identify subtask columns
            subtask_columns = [col for col in df.columns if col.startswith('subtask')]

            # Create the new column order
            new_columns_order = (
                list(df.columns[:supertask_index + 1]) +  # Columns before and including 'supertask'
                subtask_columns +                        # Subtask columns
                [col for col in df.columns if col not in list(df.columns[:supertask_index + 1]) + subtask_columns]  # Other columns
            )

            # Reorder the DataFrame
            return df[new_columns_order]

        def compute_accumulated_duration(self, df):
            # Define the columns to consider
            subtask_columns = [col for col in df.columns if col.startswith('subtask')]
            task_columns = ['supertask'] + subtask_columns

            # Fill NaN values with 'None' for uniformity
            df[task_columns] = df[task_columns].fillna('None')

            # Function to accumulate durations
            def accumulate_durations(row):
                tasks = [row[col] for col in task_columns if row[col] != 'None']
                duration = row['dur']
                accumulated = []
                for i in range(len(tasks)):
                    accumulated.append((tuple(tasks[:i+1]), duration))
                return accumulated

            # Apply the function to each row and create a list of tuples
            # .explode() here takes a df with a single column [(task_name, same_dur) ...] where 
            # same_dur is the same for each tuple.
            # .explode() then turns this into a df with a single column but multiple rows per tuple
            # ((task_name ...), same_dur) such that [(Task1, 0.5), (Task2, 0.5)] turns into two rows
            # (Task1, 0.5) and (Task1, Task2, 0.5)

            # accumulated_data = df.apply(accumulate_durations, axis=1)
            # accumulated_data.to_csv('accumulated_data.csv')
            accumulated_data = df.apply(accumulate_durations, axis=1).explode()
            # accumulated_data.to_csv('accumulated_data_exploded.csv')

            # Create a new DataFrame from the accumulated data
            accumulated_df = pd.DataFrame(accumulated_data.tolist(), columns=['task', 'duration'])

            # Aggregate the durations based on the task hierarchy
            result_df = accumulated_df.groupby('task', as_index=False).agg({'duration': 'sum'})

            # Split the task tuple back into separate columns
            task_df = pd.DataFrame(result_df['task'].tolist(), columns=['supertask'] + [f'subtask{i}' for i in range(len(task_columns) - 1)])
            result_df = pd.concat([task_df, result_df['duration']], axis=1)

            # Remove columns that are entirely NaN
            result_df = result_df.dropna(axis=1, how='all')

            return result_df

        @staticmethod
        def calc_accumulation_duration_df_task_branch_length(df: pd.DataFrame):
            # Define the columns to consider
            subtask_columns = [col for col in df.columns if col.startswith('subtask')]
            task_columns = ['supertask'] + subtask_columns

            # Calculate the branch length
            df['branch_length'] = df[task_columns].notna().sum(axis=1)
            
            return df

        @staticmethod
        def display_accumulated_duration_df(df: pd.DataFrame):
            cls = TaskEventReporting.TreemapTasksVisualizer

            # Define the columns to consider
            subtask_columns = [col for col in df.columns if col.startswith('subtask')]
            task_columns = ['supertask'] + subtask_columns

            df = cls.calc_accumulation_duration_df_task_branch_length(df)
            df.to_csv('df_test.csv')

            path = 'time_report.txt'
            prev_len = 0

            with open(path, 'w') as f:
                for _, row in df.iterrows():
                    len = row['branch_length']
                    num_spaces = 2 * len * ' '
                    dur = round(row['duration'] * 100) / 100

                    hh = f'{int(dur):02}'
                    mm = f'{int(60 * (dur - int(dur))):02}'

                    # just to add some spacing, it's the second-layer that tends to be most dense
                    # in my experience.
                    if prev_len != 1 and len == 2:
                        f.write('\n')

                    if len == 1:
                        last_task = row['supertask']
                    else:
                        last_task = row[f'subtask{len - 2}']
                    f.write(f'{num_spaces}{hh}:{mm} {last_task}\n')
                    prev_len = len

            
            print(f'Wrote report to {path}')
            

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

        print(df)
        print(df.shape)

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
        TaskEventReporting.TreemapTasksVisualizer(args).visualize()


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
                                      help="Visualize time taken to execute tasks")
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
    subparser.add_argument("--text-only", action="store_true", help="This instead reports a textual summary")
    subparser.add_argument("--priority", type=str, choices=['unimportant', 'normal'], default='unimportant', help="Set priority")
    subparser.add_argument("--treemap-color", type=str, choices=['week', 'hour'], default='week', help="Set treemap color")

    # Parse the arguments
    args = parser.parse_args()
    print(args)

    return args


if __name__ == "__main__":
    main(parse_args())
