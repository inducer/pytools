from pytools.tag import Tag, UniqueTag


class FruitNameTag(UniqueTag):
    def __init__(self, tag):
        self.tag = tag


class MangoTag(Tag):
    def __str__(self):
        return "mango"


class AppleTag(Tag):
    def __str__(self):
        return "apple"


class ColorTag(Tag):
    def __init__(self, color):
        self.color = color


class LocalAxisTag(Tag):
    def __init__(self, axis):
        self.axis = axis
