import datetime
import json
from dataclasses import asdict, is_dataclass

from dateutil import parser


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        if isinstance(o, datetime.datetime):
            return {
                '_type': 'datetime',
                'value': o.isoformat()
            }
        return super().default(o)
    
    
class EnhancedJSONDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj: dict):
        if '_type' not in obj:
            return obj
        type = obj['_type']
        if type == 'datetime':
            return parser.parse(obj['value'])
        return obj
