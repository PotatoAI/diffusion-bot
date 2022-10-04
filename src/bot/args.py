import re
import random
import sys
from pydantic import BaseModel
from typing import Optional


def gen_seed() -> int:
    return random.randint(-sys.maxsize, sys.maxsize)


class GenerateArgs(BaseModel):
    prompt: str
    seed: Optional[int]
    seedwalk: int = 0
    count: int = 1

    def sanity_check(self):
        assert len(self.prompt) > 0, 'Prompt can\'t be empty'
        assert self.count > 0, 'Count has to be greater than 0'
        assert self.count <= 10, 'Count has to be lower than 10'
        if self.count > 1:
            assert self.seed == None, 'Cant specify seed if count is greater than 1'

    @classmethod
    def from_prompt(cls, prompt: str, command: str = '/gen'):
        args_list = ['seed', 'count', 'seedwalk']
        arguments = dict()

        for arg in args_list:
            p = re.compile(f'{arg}=(-?[0-9]+)')
            m = p.search(prompt)
            if m:
                arg_str = m.group(1)
                value = int(arg_str)
                arguments[arg] = value
                prompt = p.sub('', prompt).strip()

        arguments['prompt'] = prompt.replace("/gen", "").strip()

        return cls(**arguments)
