from dataclasses import dataclass
from typing import Iterator
from dataclasses_json import dataclass_json

from pysrc.models.column_information import ColumnInformation
from pysrc.models.ind import IND
from pysrc.utils.dataclass_json import DataclassJson


@dataclass_json
@dataclass(frozen=True)
class MetanomeRunResults(DataclassJson):
    inds: list[IND]

    def has_ind(self, other_ind: IND) -> bool:
        """This checks whether this object has an IND that is identical to the passed-in one,
        i.e. whether they are from the same table and column, but may differ by name"""
        # Check whether it's directly contained
        if other_ind in self.inds:
            return True
        clean_other_ind = IND(
            dependents=[
                ColumnInformation(table_name=column.table_name.split('__')[0], column_name=column.column_name)
                for column
                in other_ind.dependents
            ], referenced=[
                ColumnInformation(table_name=column.table_name.split('__')[0], column_name=column.column_name)
                for column
                in other_ind.referenced
            ])
        # Check whether cleaned other is directly contained
        if clean_other_ind in self.inds:
            return True

        clean_inds = [
            IND(
                dependents=[
                    ColumnInformation(table_name=column.table_name.split('__')[0], column_name=column.column_name)
                    for column
                    in ind.dependents
                ], referenced=[
                    ColumnInformation(table_name=column.table_name.split('__')[0], column_name=column.column_name)
                    for column
                    in ind.referenced
                ])
            for ind
            in self.inds
        ]
        # Check whether cleaned version is in cleaned version
        return clean_other_ind in clean_inds

    def __len__(self) -> int:
        return len(self.inds)

    def __iter__(self) -> Iterator[IND]:
        return self.inds.__iter__()

    def __hash__(self) -> int:
        return hash((tuple(self.inds)))
