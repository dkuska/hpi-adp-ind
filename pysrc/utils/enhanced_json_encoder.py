from dataclasses import asdict, is_dataclass
import datetime
import json


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        if isinstance(o, datetime.date):
            return o.isoformat()
        return super().default(o)
