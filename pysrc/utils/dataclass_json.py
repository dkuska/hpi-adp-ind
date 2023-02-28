from typing import Generic, Type, TypeVar, Union, overload


JsonData = Union[str, bytes, bytearray]
Json = Union[dict, list, str, int, float, bool, None]
T = TypeVar('T')


class DataclassJsonSchema(Generic[T]):
    def dumps(self, obj: list[T], many: bool | None = None, *args, **kwargs) -> str: ... # type: ignore[empty-body]

    def loads(self, json_data: JsonData, many: bool = True, partial: bool | None = None, unknown: str | None = None, **kwargs) -> list[T]: ... # type: ignore[empty-body]


"""A class that just contains the types for the methods added by `dataclass_json`"""
class DataclassJson:
    def to_json(self, **kwargs) -> str: ... # type: ignore[empty-body]

    @classmethod
    def from_json(cls: Type[T], s: JsonData, **kwargs) -> T: ... # type: ignore[empty-body]

    @classmethod
    def from_dict(cls: Type[T], kvs: Json, **kwargs) -> T: ... # type: ignore[empty-body]

    def to_dict(self, encoding_json=False) -> dict[str, Json]: ... # type: ignore[empty-body]

    @classmethod
    def schema(cls: Type[T], **kwargs) -> DataclassJsonSchema: ... # type: ignore[empty-body]
