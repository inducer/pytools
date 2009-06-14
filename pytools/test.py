try:
    from py.test import mark as mark_test
except ImportError:
    def mark_test(**kwargs):
        def dec(f):
            return f
        return dec
