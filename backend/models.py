from __future__ import annotations
from typing import List
from pydantic import BaseModel, validator


class EditRequest(BaseModel):
    annotation_id: int
    new_category_id: int
    attributes: dict = {}


class DeleteRequest(BaseModel):
    annotation_id: int


class AddRequest(BaseModel):
    image_id: int
    segmentation: List[List[float]]
    category_id: int
    attributes: dict = {}

    @validator("segmentation")
    def seg_not_empty(cls, v):
        if not v or not v[0] or len(v[0]) < 6:
            raise ValueError("segmentation must have at least 3 points")
        return v


class ApproveRequest(BaseModel):
    batch_idx: int
