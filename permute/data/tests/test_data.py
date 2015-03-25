import permute.data as data
from numpy.testing import assert_equal


def test_botulinum():
    """ Test that "Botulinum" data can be loaded. """
    botulinum = data.botulinum()
    assert_equal((botulinum.size, len(botulinum.dtype)), (80, 28))


def test_chrom17m():
    """ Test that "chrom17m" data can be loaded. """
    chrom17m = data.chrom17m()
    assert_equal((chrom17m.size, len(chrom17m.dtype)), (10, 3))


def test_confocal():
    """ Test that "confocal" data can be loaded. """
    confocal = data.confocal()
    assert_equal((confocal.size, len(confocal.dtype)), (112, 17))


def test_germina():
    """ Test that "germina" data can be loaded. """
    germina = data.germina()
    assert_equal((germina.size, len(germina.dtype)), (40, 5))


def test_kenya():
    """ Test that "Kenya" data can be loaded. """
    kenya = data.kenya()
    assert_equal((kenya.size, len(kenya.dtype)), (16, 3))


def test_massaro_blair():
    """ Test that "massaro_blair" data can be loaded. """
    massaro_blair = data.massaro_blair()
    assert_equal((massaro_blair.size, len(massaro_blair.dtype)), (29, 2))


def test_monachus():
    """ Test that "monachus" data can be loaded. """
    monachus = data.monachus()
    assert_equal(monachus.size, 12)
    assert_equal(len(monachus.dtype), 17)


def test_mult():
    """ Test that "mult" data can be loaded. """
    mult = data.mult()
    assert_equal(mult.size, 16)
    assert_equal(len(mult.dtype), 4)


def test_perch():
    """ Test that "perch" data can be loaded. """
    perch = data.perch()
    assert_equal(perch.size, 108)
    assert_equal(len(perch.dtype), 31)


def test_rats():
    """ Test that "rats" data can be loaded. """
    rats = data.rats()
    assert_equal(rats.size, 36)
    assert_equal(len(rats.dtype), 19)


def test_setig():
    """ Test that "setig" data can be loaded. """
    setig = data.setig()
    assert_equal(setig.size, 334)
    assert_equal(len(setig.dtype), 6)


def test_urology():
    """ Test that "urology" data can be loaded. """
    urology = data.urology()
    assert_equal(urology.size, 481)
    assert_equal(len(urology.dtype), 31)


def test_washing_test():
    """ Test that "washing_test" data can be loaded. """
    washing_test = data.washing_test()
    assert_equal(washing_test.size, 800)
    assert_equal(len(washing_test.dtype), 4)


def test_waterfalls():
    """ Test that "waterfalls" data can be loaded. """
    waterfalls = data.waterfalls()
    assert_equal(waterfalls.size, 42)
    assert_equal(len(waterfalls.dtype), 17)
