from bot.args import GenerateArgs


def test_args_seed():
    a = GenerateArgs.from_prompt('/gen some text seed=69')
    assert (a.prompt == 'some text')
    assert (a.seed == 69)
    assert (a.count == 1)
    assert (a.seedwalk == 0)


def test_args_seed_and_count():
    a = GenerateArgs.from_prompt('/gen some text seed=69 count=21')
    assert (a.prompt == 'some text')
    assert (a.seed == 69)
    assert (a.count == 21)
    assert (a.seedwalk == 0)


def test_args_seed_and_count_reverse():
    a = GenerateArgs.from_prompt('/gen some text count=68 seed=23')
    assert (a.prompt == 'some text')
    assert (a.count == 68)
    assert (a.seed == 23)
    assert (a.seedwalk == 0)


def test_args_seed_and_seedwalk():
    a = GenerateArgs.from_prompt('/gen some text seedwalk=68 seed=23')
    assert (a.prompt == 'some text')
    assert (a.seed == 23)
    assert (a.seedwalk == 68)
