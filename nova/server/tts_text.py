"""Подготовка текста к синтезу речи.

TTS-модели коверкают цифры, латиницу и символы — переводим их в
произносимый русский. Меняется только текст для синтезатора;
то, что показывается пользователю, остаётся как есть.
"""

import re

from num2words import num2words

# эмоциональные ремарки fish s2 вида [sarcastic], [soft tone] — латиница
# в квадратных скобках; уходят в синтез, но не на экран
_MARKER_RE = re.compile(r"\[[a-z][a-z '\-]*\]")


def strip_markers(text: str) -> str:
    """Текст для экрана/истории — без голосовых ремарок."""
    return re.sub(r"\s{2,}", " ", _MARKER_RE.sub("", text)).strip()


# звуковые ремарки (не эмоции-инструкции): смешок «для разгона» в начале
# реплики звучит как нервный тик — модель игнорирует правило в промпте,
# поэтому вырезаем кодом
_SOUND_TAGS = (
    "laughing", "chuckling", "sobbing", "crying loudly", "sighing",
    "groaning", "panting", "gasping", "yawning", "snoring",
    "audience laughing", "background laughter", "crowd laughing",
    "break", "long-break",
)
_LEADING_SOUNDS_RE = re.compile(
    r"^\s*(?:\[(?:" + "|".join(re.escape(t) for t in _SOUND_TAGS) + r")\]\s*)+")


def drop_leading_sounds(text: str) -> str:
    """Убрать звуковые ремарки до первого слова; в середине/конце — можно."""
    return _LEADING_SOUNDS_RE.sub("", text)


# типовые галлюцинации виспера на бормотании/тишине: модель выдаёт
# ютуб-титры вместо честного «не разобрал»
_ASR_GARBAGE_RE = re.compile(
    r"субтитр|dimatorzok|продолжение следует|спасибо за просмотр|"
    r"редактор субтитров|подписывайтесь",
    re.IGNORECASE,
)


def asr_garbage(text: str) -> bool:
    """Похоже ли распознанное на галлюцинацию ASR, а не на речь."""
    return bool(_ASR_GARBAGE_RE.search(text)) or not text.strip()

# символы -> слова (или пробел, если озвучивать нечего)
_SYMBOLS = {
    "%": " процентов",
    "№": " номер ",
    "$": " долларов",
    "€": " евро",
    "+": " плюс ",
    "=": " равно ",
    "&": " и ",
    "#": " решётка ",
    "@": " собака ",
    "<": " меньше ",
    ">": " больше ",
    "*": " ",
    "`": " ",
    "_": " ",
    "~": " ",
    "^": " ",
    "|": " ",
    "/": " ",
    "\\": " ",
}

# частые слова — руками: транслит по буквам их коверкает
_WORDS = {
    "ok": "окей", "hi": "хай", "gg": "джиджи", "wp": "вэпэ",
    "windows": "виндоус", "discord": "дискорд", "minecraft": "майнкрафт",
    "python": "пайтон", "powershell": "пауэршелл", "google": "гугл",
    "youtube": "ютуб", "twitch": "твич", "steam": "стим", "github": "гитхаб",
    "wifi": "вайфай", "online": "онлайн", "update": "апдейт",
    "chrome": "хром", "telegram": "телеграм", "nvidia": "энвидиа",
    "fps": "эфпээс", "gpu": "джипию", "cpu": "ципию", "hp": "хэпэ",
    "pc": "писи", "ai": "эйай", "vs": "версус",
}

# диграфы и буквы для запасного транслита незнакомых слов
_DIGRAPHS = [
    ("sch", "ск"), ("tch", "ч"), ("sh", "ш"), ("ch", "ч"), ("th", "т"),
    ("ph", "ф"), ("wh", "в"), ("ck", "к"), ("oo", "у"), ("ee", "и"),
    ("ea", "и"), ("ay", "ей"), ("ai", "ей"), ("ou", "ау"), ("qu", "кв"),
    ("kn", "н"), ("ng", "нг"),
]
_LETTERS = {
    "a": "а", "b": "б", "c": "к", "d": "д", "e": "е", "f": "ф", "g": "г",
    "h": "х", "i": "и", "j": "дж", "k": "к", "l": "л", "m": "м", "n": "н",
    "o": "о", "p": "п", "q": "к", "r": "р", "s": "с", "t": "т", "u": "у",
    "v": "в", "w": "в", "x": "кс", "y": "й", "z": "з",
}


def _translit(word: str) -> str:
    w = word.lower()
    for digraph, repl in _DIGRAPHS:
        w = w.replace(digraph, repl)
    return "".join(_LETTERS.get(ch, ch) for ch in w)


def _latin_word(m: re.Match) -> str:
    word = m.group(0)
    return _WORDS.get(word.lower(), _translit(word))


def _number(m: re.Match) -> str:
    token = m.group(0).replace(",", ".")
    try:
        if "." in token:
            return num2words(float(token), lang="ru")
        return num2words(int(token), lang="ru")
    except (ValueError, OverflowError):
        return token


def normalize_for_tts(text: str) -> str:
    # спрятать ремарки, чтобы транслитерация не превратила [sarcastic]
    # в [саркастик]; плейсхолдер без цифр и латиницы — его не тронут
    # ни числовая замена, ни транслит
    markers: list[str] = []

    def _hide(m: re.Match) -> str:
        markers.append(m.group(0))
        return "\x00" + "ъ" * len(markers) + "\x00"

    out = _MARKER_RE.sub(_hide, text)
    # время вида 2:30 — «два тридцать»
    out = re.sub(
        r"\b(\d{1,2}):(\d{2})\b",
        lambda m: f"{num2words(int(m.group(1)), lang='ru')} "
                  f"{num2words(int(m.group(2)), lang='ru')}",
        out,
    )
    # числа, включая десятичные с точкой или запятой между цифрами
    out = re.sub(r"\d+[.,]\d+|\d+", _number, out)
    # латинские слова
    out = re.sub(r"[A-Za-z]+", _latin_word, out)
    for sym, repl in _SYMBOLS.items():
        out = out.replace(sym, repl)
    out = re.sub(r"\x00(ъ+)\x00", lambda m: markers[len(m.group(1)) - 1], out)
    # подчистить множественные пробелы
    return re.sub(r"\s{2,}", " ", out).strip()
