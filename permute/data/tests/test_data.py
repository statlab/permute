import permute.data as data
from numpy.testing import assert_equal

def test_botulinum():
    """ Test that "Botulinum" data can be loaded. """
    botulinum = data.botulinum()
    assert_equal(botulinum.shape, (80, 28))

def test_chrom17m():
    """ Test that "chrom17m" data can be loaded. """
    chrom17m = data.chrom17m()
    assert_equal(chrom17m.shape, (10, 3))

def test_confocal():
    """ Test that "confocal" data can be loaded. """
    confocal = data.confocal()
    assert_equal(confocal.shape, (112, 17))

def test_germina():
    """ Test that "germina" data can be loaded. """
    germina = data.germina()
    assert_equal(germina.shape, (40, 5))

def test_kenya():
    """ Test that "Kenya" data can be loaded. """
    kenya = data.kenya()
    assert_equal(kenya.shape, (16, 3))

def test_massaro_blair():
    """ Test that "massaro_blair" data can be loaded. """
    massaro_blair = data.massaro_blair()
    assert_equal(massaro_blair.shape, (29, 2))

def test_monachus():
    """ Test that "monachus" data can be loaded. """
    monachus = data.monachus()
    assert_equal(monachus.shape, (12, 17))

def test_mult():
    """ Test that "mult" data can be loaded. """
    mult = data.mult()
    assert_equal(mult.shape, (16, 4))

def test_perch():
    """ Test that "perch" data can be loaded. """
    perch = data.perch()
    assert_equal(perch.shape, (108, 31))

def test_rats():
    """ Test that "rats" data can be loaded. """
    rats = data.rats()
    assert_equal(rats.shape, (36, 19))

def test_setig():
    """ Test that "setig" data can be loaded. """
    setig = data.setig()
    assert_equal(setig.shape, (334, 6))

def test_urology():
    """ Test that "urology" data can be loaded. """
    urology = data.urology()
    assert_equal(urology.shape, (481, 31))

def test_washing_test():
    """ Test that "washing_test" data can be loaded. """
    washing_test = data.washing_test()
    assert_equal(washing_test.shape, (800, 4))

def test_waterfalls():
    """ Test that "waterfalls" data can be loaded. """
    waterfalls = data.waterfalls()
    assert_equal(waterfalls.shape, (42, 17))
