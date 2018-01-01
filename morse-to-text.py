#!/usr/bin/python
import csv

import numpy
import scipy.io.wavfile as wavfile
from matplotlib.pyplot import *


class Plotter:
    @staticmethod
    def saveplot(name, data, length=-1, height=-1, dpi=None):
        plot(data)
        if length != -1:
            axis(xmax=length)
        if height != -1:
            axis(ymax=height)
        # savefig(name + "." + self.format, format=self.format, dpi=dpi)
        cla()

    @staticmethod
    def specgram(name, signal):
        spectrogram = specgram(signal, Fs=2)
        # savefig(name + "." + self.format, format=self.format)
        cla()
        return spectrogram


class DummyPlotter:

    def saveplot(self, name, data, length=-1, height=-1, dpi=None):
        return None

    def specgram(self, name, signal):
        spectrogram = specgram(signal)
        cla()
        return spectrogram


class AudioArray:
    def __init__(self, array: numpy.ndarray):
        self.data = array

    def setdata(self, data):
        self.data = data

    def getdata(self):
        return self.data

    def getlength(self):
        return len(self.data)

    def saveas(self, path):
        numpy.savez_compressed(path, self.data)

class SoundFile:

    def __init__(self, path):
        # 1 - leer el archivo con las muestras
        #	el resultado de read es una tupla, el elemento 1 tiene las muestras
        the_file = wavfile.read(path)
        self.rate = the_file[0]
        self.length = len(the_file[1])
        self.data = the_file[1]
        # appendea ceros hasta completar una potencia de 2
        power = 10
        while pow(2, power) < self.length:
            power += 1
        self.data = numpy.append(self.data, numpy.zeros(pow(2, power) - self.length))

    def setdata(self, data):
        self.data = data

    def getdata(self):
        return self.data

    def getlength(self):
        return self.length

    def saveas(self, path):
        wavfile.write(path, self.rate, self.data)


class SignalFilter:

    def filter(self, soundfile):
        # 2 - aplico transformada de fourier
        trans = numpy.fft.rfft(soundfile.getdata())
        trans_real = numpy.abs(trans)
        # 2b - lo grafico
        # plotter.saveplot("transformed", trans_real)
        # 3 - busco la frecuencia
        band = 2000
        # ignore the first 200Hz
        hzignored = 200
        frec = hzignored + numpy.argmax(trans_real[hzignored:])
        # print max(trans_real)
        # print trans_real[frec]
        # print frec
        min = (frec - band / 2) if (frec > band / 2) else 0
        filter_array = numpy.append(numpy.zeros(int(min), dtype=numpy.float64), numpy.ones(band, dtype=numpy.float64))
        filter_array = numpy.append(filter_array, numpy.zeros(len(trans_real) - len(filter_array)))
        filtered_array = numpy.multiply(trans, filter_array)
        # plotter.saveplot("filtered_trans", numpy.abs(filtered_array))
        # 4 - antitransformo
        filtered_signal = numpy.array(numpy.fft.irfft(filtered_array)[:soundfile.getlength()], dtype="int16")
        # plotter.saveplot("filtered_signal", filtered_signal)
        soundfile.setdata(filtered_signal)


class SpectreAnalyzer:

    def spectrogram(self, signal):
        # spectrogram = specgram(signal)
        # savefig("spectrogram", format="pdf")
        # cla()
        spectrogram = Plotter().specgram("spectrogram", signal)
        return spectrogram

    def sumarizecolumns(self, mat):
        vec_ones = numpy.ones(len(mat))
        vec_sum = (numpy.matrix(vec_ones) * numpy.matrix(mat)).transpose()
        # plotter.saveplot("frecuency_volume", vec_sum)
        return vec_sum

    def findpresence(self, vec_sum):
        presence = numpy.zeros(len(vec_sum))
        threshold = numpy.max(vec_sum) / 2.0
        for i in range(len(presence)):
            if vec_sum[i] > threshold:
                presence[i] = 1
        # plotter.saveplot("presence", presence, dpi=300, height=5)
        return presence

    def findpulses(self, soundfile):
        spec = self.spectrogram(soundfile.getdata())
        # spec[0] es la matriz del rojo
        red_matrix = spec[0]
        vec_sum = self.sumarizecolumns(red_matrix)
        presence = self.findpresence(vec_sum)
        return presence


class ShortLong:
    def __init__(self, shorts, longs):
        self.shortmean = numpy.mean(shorts)
        self.shortstd = numpy.std(shorts)
        self.longmean = numpy.mean(longs)
        self.longstd = numpy.std(longs)

    def tostring(self):
        return "short: (" + repr(self.shortmean) + ", " + repr(self.shortstd) + ")\n\
long: (" + repr(self.longmean) + ", " + repr(self.longstd) + ")"


class PulsesAnalyzer:

    def compress(self, pulses):
        vec = []
        i = 0

        if pulses[0] == 1:
            vec += [0]
            i = 1

        last = pulses[0]

        while i < len(pulses):
            c = 0
            last = pulses[i]
            while i < len(pulses) and pulses[i] == last:
                i += 1
                c += 1
            vec += [c]
            i += 1

        vec = vec[1:-1]
        return vec

    def split(self, vec):
        onesl = numpy.zeros(1 + len(vec) // 2)
        zerosl = numpy.zeros(len(vec) // 2)
        for i in range(len(vec) // 2):
            onesl[i] = vec[2 * i]
            zerosl[i] = vec[2 * i + 1]
        onesl[-1] = vec[-1]
        return onesl, zerosl

    def findshortlongdup(self, vec):
        sor = numpy.sort(vec)
        last = sor[0]
        for i in range(len(sor))[1:]:
            if sor[i] > 2 * last:
                shorts = sor[:i - 1]
                longs = sor[i:]
                return shorts, longs
        return vec, []

    def createshortlong(self, shorts, longs):
        return ShortLong(shorts, longs)

    def findshortlong(self, vec):
        dup = self.findshortlongdup(vec)
        return self.createshortlong(dup[0], dup[1])


class SymbolDecoder:
    def __init__(self, onessl, zerossl, zeroextra=None):
        self.onessl = onessl
        self.zerossl = zerossl
        self.zeroextra = zeroextra

    def get(self, sl, n, ifshort, iflong, ifnone="?"):
        d = 4
        if (n > sl.shortmean - d * sl.shortstd) and (n < sl.shortmean + d * sl.shortstd):
            return ifshort
        if (n > sl.longmean - d * sl.longstd) and (n < sl.longmean + d * sl.longstd):
            return iflong
        return ifnone

    def getonesymbol(self, n):
        return self.get(self.onessl, n, ".", "-")

    def getzerosymbol(self, n):
        sym = self.get(self.zerossl, n, "", " ")
        if sym == "":
            return sym
        return self.get(self.zeroextra, n, " ", " | ", ifnone=" ")


class PulsesTranslator:
    def tostring(self, pulses):
        pa = PulsesAnalyzer()
        comp_vec = pa.compress(pulses)
        comp_tup = pa.split(comp_vec)

        onessl = pa.findshortlong(comp_tup[0])
        # zeros are subdivided
        dup = pa.findshortlongdup(comp_tup[1])
        zerossl = pa.createshortlong(dup[0], dup[1])
        dup2 = pa.findshortlongdup(dup[1])
        zeroextra = pa.createshortlong(dup2[0], dup2[1])

        symdec = SymbolDecoder(onessl, zerossl, zeroextra)

        s = ""
        for i in range(len(comp_vec) // 2):
            s += symdec.getonesymbol(comp_vec[2 * i])
            s += symdec.getzerosymbol(comp_vec[2 * i + 1])
        s += symdec.getonesymbol(comp_vec[-1])
        return s


class Codes:
    def __init__(self, path):
        data = csv.DictReader(open(path), delimiter=',', fieldnames=["char", "code"])
        self.dic = {}
        for entry in data:
            self.dic[entry["code"]] = entry["char"]

    def tochar(self, code):
        if code in self.dic:
            return self.dic[code]
        return "?"


class StringTranslator:
    def __init__(self):
        self.codes = Codes("codes.csv")

    def totext(self, s):
        text = ""
        for code in s.split():
            if code == "|":
                char = " "
            else:
                char = self.codes.tochar(code)
            text += char
        return text


def decode_morse_audio(path: str):
    the_file = SoundFile(path)
    # the_file.saveplot("original")

    the_filter = SignalFilter()
    the_filter.filter(the_file)
    # the_file.saveas("filtered.wav")

    analyzer = SpectreAnalyzer()
    pulses = analyzer.findpulses(the_file)

    pul_translator = PulsesTranslator()
    code_string = pul_translator.tostring(pulses)

    str_translator = StringTranslator()
    s = str_translator.totext(code_string)

    return s


if __name__ == '__main__':
    print(decode_morse_audio('az.wav'))
