import collections


class LazyDict(collections.abc.Mapping):
    def __init__(self, seq=None, *, constructor, initializer=None, **kwargs):
        self.__items = dict(seq or (), **kwargs)
        self.__constructor = constructor
        self.__initializer = initializer or (lambda x: None)

    def __len__(self) -> int:
        return len(self.__items)

    def __iter__(self):
        return iter(self.__iter__())

    def __getitem__(self, item):
        try:
            return self.__items[item]
        except KeyError:
            value = self.__constructor(item)
            if value is None:
                raise KeyError(item)
            self.__items[item] = value
            self.__initializer(item)
            return value

    def __contains__(self, item):
        return item in self.__items


class LazyEmptyDict(collections.abc.MutableMapping):
    def __init__(self, seq=None, *, constructor, initializer=None, **kwargs):
        self.__items = dict(seq or (), **kwargs)
        self.__constructor = constructor
        self.__initializer = initializer or (lambda x: None)

    def __len__(self) -> int:
        return len(self.__items)

    def __iter__(self):
        return iter(self.__iter__())

    def __getitem__(self, item):
        try:
            return self.__items[item]
        except KeyError:
            value = self.__constructor(item)
            if value is None:
                return value

            self.__items[item] = value
            self.__initializer(item)
            return value

    def __setitem__(self, key, value):
        self.__items[key] = value

    def __delitem__(self, key) -> None:
        del self.__items[key]

    def __contains__(self, item):
        return item in self.__items
