import six
import functools

class KoreanHandler:
    
    def __init__(self):
        self.TYPE_INITIAL = 0x001
        self.TYPE_MEDIAL = 0x010
        self.TYPE_FINAL = 0x100
        self.INITIALS = list(map(six.unichr, [0x3131, 0x3132, 0x3134, 0x3137, 0x3138, 0x3139,
                                  0x3141, 0x3142, 0x3143, 0x3145, 0x3146, 0x3147,
                                  0x3148, 0x3149, 0x314a, 0x314b, 0x314c, 0x314d,
                                  0x314e]))
        self.MEDIALS = list(map(six.unichr, [0x314f, 0x3150, 0x3151, 0x3152, 0x3153, 0x3154,
                                 0x3155, 0x3156, 0x3157, 0x3158, 0x3159, 0x315a,
                                 0x315b, 0x315c, 0x315d, 0x315e, 0x315f, 0x3160,
                                 0x3161, 0x3162, 0x3163]))
        self.FINALS = list(map(six.unichr, [0x3131, 0x3132, 0x3133, 0x3134, 0x3135, 0x3136,
                                0x3137, 0x3139, 0x313a, 0x313b, 0x313c, 0x313d,
                                0x313e, 0x313f, 0x3140, 0x3141, 0x3142, 0x3144,
                                0x3145, 0x3146, 0x3147, 0x3148, 0x314a, 0x314b,
                                0x314c, 0x314d, 0x314e]))
        self.CHAR_LISTS = {self.TYPE_INITIAL: self.INITIALS,
                           self.TYPE_MEDIAL: self.MEDIALS,
                           self.TYPE_FINAL: self.FINALS}
        self.CHAR_SETS = dict(map(lambda x: (x[0], set(x[1])), six.iteritems(self.CHAR_LISTS)))
        self.CHARSET = functools.reduce(lambda x, y: x.union(y), self.CHAR_SETS.values(), set())
        self.CHAR_INDICES = dict(map(lambda x: (x[0], dict([(c, i) for i, c in enumerate(x[1])])),
                six.iteritems(self.CHAR_LISTS)))

        def check_syllable(self, x):
            return 0xAC00 <= ord(x) <= 0xD7A3

        def jamo_type(self, x):
            t = 0x000
            for type_code, jamo_set in six.iteritems(self.CHAR_SETS):
                if x in jamo_set:
                    t |= type_code
            
            return t

        def split_syllable_char(self, x):
            """
            Splits a given korean character into components.
            """
            if len(x) != 1:
                raise ValueError("Input string must have exactly one character.")

            if not check_syllable(x):
                raise ValueError(
                    "Input string does not contain a valid Korean character.")

            diff = ord(x) - 0xAC00
            _m = diff % 28
            _d = (diff - _m) // 28

            initial_index = _d // 21
            medial_index = _d % 21
            final_index = _m

            if not final_index:
                result = (self.INITIALS[initial_index], self.MEDIALS[medial_index])
            else:
                result = (
                    self.INITIALS[initial_index], self.MEDIALS[medial_index],
                    self.FINALS[final_index - 1])

            return result

        def join_jamos_char(self, initial, medial, final=None):
            """
            Combines jamos to produce a single syllable.
            """
            if initial not in self.CHAR_SETS[self.TYPE_INITIAL] or medial not in self.CHAR_SETS[
                self.TYPE_MEDIAL] or (final and final not in self.CHAR_SETS[self.TYPE_FINAL]):
                raise ValueError("Given Jamo characters are not valid.")

            initial_ind = self.CHAR_INDICES[self.TYPE_INITIAL][initial]
            medial_ind = self.CHAR_INDICES[self.TYPE_MEDIAL][medial]
            final_ind = self.CHAR_INDICES[self.TYPE_FINAL][final] + 1 if final else 0

            b = 0xAC00 + 28 * 21 * initial_ind + 28 * medial_ind + final_ind

            result = six.unichr(b)

            assert self.check_syllable(result)

            return result

        def split_syllables(self, string):
            """
            Splits a sequence of Korean syllables to produce a sequence of jamos.
            Irrelevant characters will be ignored.
            """
            new_string = ""
            for c in string:
                if not check_syllable(c):
                    new_c = c
                else:
                    new_c = "".join(split_syllable_char(c))
                new_string += new_c

            return new_string

        def join_jamos(self, string):
            """
            Combines a sequence of jamos to produce a sequence of syllables.
            Irrelevant characters will be ignored.
            """
            last_t = 0
            queue = []
            new_string = ""

            def flush(queue, n=0):
                new_queue = []

                while len(queue) > n:
                    new_queue.append(queue.pop())

                if len(new_queue) == 1:
                    result = new_queue[0]
                elif len(new_queue) >= 2:
                    try:
                        result = join_jamos_char(*new_queue)
                    except ValueError:
                        # Invalid jamo combination
                        result = "".join(new_queue)
                else:
                    result = None

                return result

            for c in string:
                if c not in self.CHARSET:
                    if queue:
                        new_c = flush(queue) + c
                    else:
                        new_c = c

                    last_t = 0
                else:
                    t = jamo_type(c)
                    new_c = None

                    if t & self.TYPE_FINAL == self.TYPE_FINAL:
                        if not (last_t == self.TYPE_MEDIAL):
                            new_c = flush(queue)
                    elif t == self.TYPE_INITIAL:
                        new_c = flush(queue)
                    elif t == self.TYPE_MEDIAL:
                        if last_t & self.TYPE_INITIAL == self.TYPE_INITIAL:
                            new_c = flush(queue, 1)
                        else:
                            new_c = flush(queue)

                    last_t = t
                    queue.insert(0, c)

                if new_c:
                    new_string += new_c

            if queue:
                new_string += flush(queue)

            return new_string
