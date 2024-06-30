PART_GROUP_MAP_FINE = {
    "head": [
        "has_bill_color",
        "has_eye_color",
        "has_crown_color",
        "has_forehead_color",
        "has_bill_length",
        "has_bill_shape",
        "has_nape_color",
        "has_head_pattern",
    ],
    "breast": ["has_throat_color", "has_breast_pattern", "has_breast_color"],
    "belly": ["has_underparts_color", "has_belly_color", "has_belly_pattern"],
    "back": ["has_upperparts_color", "has_back_pattern", "has_back_color"],
    "wing": ["has_wing_pattern", "has_wing_color", "has_wing_shape"],
    "tail": ["has_tail_shape", "has_tail_pattern", "has_upper_tail_color", "has_under_tail_color"],
    "leg": ["has_leg_color"],
    "others": ["has_primary_color", "has_shape", "has_size"],
}

GROUP_PART_MAP_FINE = {}
for part, group_names in PART_GROUP_MAP_FINE.items():
    for group in group_names:
        GROUP_PART_MAP_FINE[group] = part

PART_REMAP_FINE = {
    "back": ["back"],
    "beak": ["bill", "head"],
    "belly": ["belly"],
    "breast": ["breast"],
    "crown": ["crown", "head"],
    "forehead": ["forehead","head"],
    "left eye": [
        "eye",
        "head",
    ],
    "left leg": [
        "leg",
    ],
    "left wing": [
        "wing",
    ],
    "nape": [
        "nape",
        "head",
    ],
    "right eye": ["eye", "head"],
    "right leg": [
        "leg",
    ],
    "right wing": [
        "wing",
    ],
    "tail": [
        "tail",
    ],
    "throat": [
        "throat",
    ],
}

PART_REMAP_COARSE = {
    "back": [
        "back",
    ],
    "beak": [
        "head",
    ],
    "belly": [
        "belly",
    ],
    "breast": [
        "breast",
    ],
    "crown": [
        "head",
    ],
    "forehead": [
        "head",
    ],
    "left eye": [
        "head",
    ],
    "left leg": [
        "leg",
    ],
    "left wing": [
        "wing",
    ],
    "nape": [
        "head",
    ],
    "right eye": [
        "head",
    ],
    "right leg": [
        "leg",
    ],
    "right wing": [
        "wing",
    ],
    "tail": [
        "tail",
    ],
    "throat": [
        "breast",
    ],
}
