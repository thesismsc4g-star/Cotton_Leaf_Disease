CLASS_NAMES = [
    "aug_Alternaria_Leaf",
    "aug_Bacterial_Blight",
    "aug_Fusarium_Wilt",
    "aug_Healthy_Leaf",
    "aug_Verticillium_Wilt",
]

DISPLAY_NAMES = {
    "aug_Alternaria_Leaf": "Alternaria Leaf Spot",
    "aug_Bacterial_Blight": "Bacterial Blight",
    "aug_Fusarium_Wilt": "Fusarium Wilt",
    "aug_Healthy_Leaf": "Healthy Leaf",
    "aug_Verticillium_Wilt": "Verticillium Wilt",
}

TEXT_PROMPTS = {
    "aug_Alternaria_Leaf": [
        "a photo of a cotton leaf with target-like circular spots and concentric rings",
        "a cotton leaf with brown lesions and purple borders",
        "a diseased cotton leaf showing multiple round fungal spots with ring patterns",
        "a cotton leaf with dark circular lesions and layered ring structures",
        "an infected cotton leaf with expanding brown spots and halo-like margins",
    ],
    "aug_Bacterial_Blight": [
        "a photo of a cotton leaf with angular lesions shaped by veins",
        "a cotton leaf with water-soaked spots and dark outlines",
        "a diseased cotton leaf showing vein-limited bacterial lesions",
        "a cotton leaf with irregular angular dark patches",
        "an infected cotton leaf with wet-looking lesions bounded by veins",
    ],
    "aug_Fusarium_Wilt": [
        "a photo of a cotton leaf with yellowing margins and inward drying",
        "a cotton leaf showing necrosis spreading from the edges to the center",
        "a wilted cotton leaf with vascular discoloration symptoms",
        "a cotton leaf with dry curled structure and yellow-brown edges",
        "a cotton leaf showing severe drying and tissue death",
    ],
    "aug_Healthy_Leaf": [
        "a photo of a healthy cotton leaf with bright green color and no damage",
        "a clean cotton leaf with uniform texture and visible veins",
        "a green cotton leaf without spots lesions or discoloration",
        "a fresh cotton leaf with smooth surface and natural structure",
        "an undamaged cotton leaf with consistent green pigmentation",
    ],
    "aug_Verticillium_Wilt": [
        "a photo of a cotton leaf with yellowing and drooping symptoms",
        "a wilted cotton leaf with red and brown discoloration",
        "a cotton leaf showing drying curling and vascular wilting",
        "a cotton leaf with uneven yellow patches and decay",
        "a cotton leaf with progressive wilting and necrotic edges",
    ],
}

ORDERED_TEXT_PROMPTS = {cls: TEXT_PROMPTS[cls] for cls in CLASS_NAMES}


def pretty_class_name(class_name: str) -> str:
    return DISPLAY_NAMES.get(class_name, class_name)
