import sys
import testlib  # noqa


def test_parse_tags():
    from pytools.tag import parse_tag

    def assert_same_as_python(tag_text):
        assert parse_tag(tag_text) == eval(tag_text)

    assert_same_as_python("testlib.FruitNameTag(testlib.MangoTag())")
    assert_same_as_python("testlib.LocalAxisTag(0)")
    assert_same_as_python('testlib.ColorTag("blue")')
    assert_same_as_python('testlib.ColorTag(color="blue")')
    assert_same_as_python("testlib.LocalAxisTag(axis=0)")

    assert (parse_tag("testlib.FruitNameTag(mango)",
                     shortcuts={"mango": testlib.MangoTag(),
                                "apple": testlib.AppleTag()})
            == testlib.FruitNameTag(testlib.MangoTag()))

    assert (parse_tag("l.0",
                     shortcuts={"l.0": testlib.LocalAxisTag(0)})
            == testlib.LocalAxisTag(axis=0))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
