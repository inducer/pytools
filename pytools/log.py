from warnings import warn

warn("pytools.log was moved to https://github.com/illinois-ceesd/logpyle/. "
        "I will try to import that for you. If the import fails, say "
        "'pip install logpyle', and change your imports from 'pytools.log' "
        "to 'logpyle'.", DeprecationWarning)

from logpyle import *  # noqa  # pylint: disable=import-error
