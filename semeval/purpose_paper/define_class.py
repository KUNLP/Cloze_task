from typing import Optional, List, Union
from dataclasses import dataclass

@dataclass
class InputExample:
    id: str
    title: str
    prev_sentence: str
    now_sentence: str
    next_sentence: str
    answer: str
    label: Optional[str] = None


@dataclass
class InputFeature:
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None