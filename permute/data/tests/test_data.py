import permute.data as data
from numpy.testing import assert_equal


def test_botulinum():
    """ Test that "Botulinum" data can be loaded. """
    botulinum = data.botulinum()
    assert (botulinum.size, len(botulinum.dtype)) == (80, 28)


def test_chrom17m():
    """ Test that "chrom17m" data can be loaded. """
    chrom17m = data.chrom17m()
    assert (chrom17m.size, len(chrom17m.dtype)) == (10, 3)

def test_clinical_trial():
    """ Test that "rb_clinical_trial" data can be loaded. """
    clin = data.clinical_trial()
    assert (clin.size, len(clin.dtype)) == (272, 15)


def test_confocal():
    """ Test that "confocal" data can be loaded. """
    confocal = data.confocal()
    assert (confocal.size, len(confocal.dtype)) == (112, 17)


def test_germina():
    """ Test that "germina" data can be loaded. """
    germina = data.germina()
    assert (germina.size, len(germina.dtype)) == (40, 5)


def test_kenya():
    """ Test that "Kenya" data can be loaded. """
    kenya = data.kenya()
    assert (kenya.size, len(kenya.dtype)) == (16, 3)


def test_massaro_blair():
    """ Test that "massaro_blair" data can be loaded. """
    massaro_blair = data.massaro_blair()
    assert (massaro_blair.size, len(massaro_blair.dtype)) == (29, 2)


def test_monachus():
    """ Test that "monachus" data can be loaded. """
    monachus = data.monachus()
    assert monachus.size == 12
    assert len(monachus.dtype) == 17


def test_mult():
    """ Test that "mult" data can be loaded. """
    mult = data.mult()
    assert mult.size == 16
    assert len(mult.dtype) == 4


def test_perch():
    """ Test that "perch" data can be loaded. """
    perch = data.perch()
    assert perch.size == 108
    assert len(perch.dtype) == 31


def test_rats():
    """ Test that "rats" data can be loaded. """
    rats = data.rats()
    assert rats.size == 36
    assert len(rats.dtype) == 19


def test_setig():
    """ Test that "setig" data can be loaded. """
    setig = data.setig()
    assert setig.size == 334
    assert len(setig.dtype) == 6


def test_urology():
    """ Test that "urology" data can be loaded. """
    urology = data.urology()
    assert urology.size == 481
    assert len(urology.dtype) == 31


def test_washing_test():
    """ Test that "washing_test" data can be loaded. """
    washing_test = data.washing_test()
    assert washing_test.size == 800
    assert len(washing_test.dtype) == 4


def test_waterfalls():
    """ Test that "waterfalls" data can be loaded. """
    waterfalls = data.waterfalls()
    assert waterfalls.size == 42
    assert len(waterfalls.dtype) == 17


def test_ipat():
    """ Test that "ipat" data can be loaded. """
    ipat = data.ipat()
    assert ipat.size == 20
    assert len(ipat.dtype) == 2


def test_job():
    """ Test that "job" data can be loaded. """
    job = data.job()
    assert job.size == 20
    assert len(job.dtype) == 2


def test_fly():
    """ Test that "fly" data can be loaded. """
    fly = data.fly()
    assert fly.size == 70
    assert len(fly.dtype) == 8


def test_testosterone():
    """ Test that "testosterone" data can be loaded. """
    testosterone = data.testosterone()
    assert testosterone.size == 11
    assert len(testosterone.dtype) == 5


def test_worms():
    """ Test that "worms" data can be loaded. """
    worms = data.worms()
    assert worms.size == 18
    assert len(worms.dtype) == 2
