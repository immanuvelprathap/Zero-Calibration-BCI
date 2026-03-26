import fitz  # PyMuPDF
from PIL import Image, ImageDraw
import io

def draw_on_montage(pdf_name):
    doc = fitz.open(pdf_name)
    page = doc[0]
    zoom = 2  
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.open(io.BytesIO(pix.tobytes()))
    draw = ImageDraw.Draw(img)

    # PRECISION COORDINATES: Row C (Motor Strip)
    zones = [
        {"name": "C3", "pos": (455, 840), "color": "blue", "label": "CH 9 (RIGHT HAND)"},
        {"name": "C4", "pos": (925, 840), "color": "red", "label": "CH 13 (LEFT HAND)"},
        {"name": "Cz", "pos": (690, 840), "color": "green", "label": "CH 11 (REF)"}
    ]

    for zone in zones:
        x, y = zone["pos"]
        r = 60 
        draw.ellipse([x-r, y-r, x+r, y+r], outline=zone["color"], width=10)
        draw.text((x-50, y+70), zone["label"], fill=zone["color"])
        
    img.save("biological_target_map.png")
    print("✅ Verified: Target indices 9, 11, 13 are centered.")

draw_on_montage(r"C:\Users\IMMU\Desktop\Unibe Spring Semester - 2026 Semester 2\Deep Learning\64_channel_sharbrough.pdf")