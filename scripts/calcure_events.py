import csv
import json
from typing import List
from datetime import datetime as dtdt
from datetime import timedelta

CALCURE_EVENTS_CSV_PATH = '/home/lan/.task/schedule/events.csv'
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
                "title": f"{calcure_event.gcode} {calcure_event.desc}",
                "begin": calcure_event.begin_dt.strftime('%Y/%m/%d %H:%M:%S'),
                "end": calcure_event.end_dt.strftime('%Y/%m/%d %H:%M:%S'),
                "description": f"event: {calcure_event.event_type}, uuid: {calcure_event.uuid}, {subtask_uuid_desc}proj: {calcure_event.proj}, interval: {calcure_event.interval.seconds // 3600:02d}:{(calcure_event.interval.seconds // 60) % 60:02d}",
                "background": background,
            }
            result['markers'].append(schedule_event)
            pass
    
    with open(MARKERS_JSON_PATH, 'w') as json_file:
        json_file.write(json.dumps(result))

    
# 1688965338515
# {id:1688965338515,title:AAA,begin:2023/07/09 03:30:00,end:2023/07/09 04:30:00,description:"", background: rgba(256,0,0,1)},
    pass

def ExpandCalcureEventsToOrderedSequencialEvents(day: dtdt, begin: timedelta, first_id, last_id):
    with open(CALCURE_EVENTS_CSV_PATH, 'r') as calcure_events_file:
        csv_reader = csv.reader(calcure_events_file)
        for row in csv_reader:
            calcure_event = CalcureEvent.read_csv(row)
            same_day = calcure_event.begin_dt.year == day.year and calcure_event.begin_dt.month == day.month and calcure_event.begin_dt.day == day.day
            within_id_range = int(calcure_event._id) >= first_id and int(calcure_event._id) < last_id
            if not same_day or not within_id_range:
                continue
            calcure_event.begin_dt += begin
            begin += calcure_event.interval + timedelta(minutes=10)
            print(calcure_event.to_csv())

    pass

if __name__ == '__main__':
    ConvertCalcureEventsToMarkersJson()
    ExpandCalcureEventsToOrderedSequencialEvents(dtdt.strptime('2023-07-11', '%Y-%m-%d'), timedelta(hours=13, minutes=0), 240, 245)
