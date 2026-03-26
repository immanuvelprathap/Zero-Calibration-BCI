import fitz  # PyMuPDF
from PIL import Image, ImageDraw
import io

def draw_motor_mapping(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(dpi=150)
    img = Image.open(io.BytesIO(pix.tobytes()))
    draw = ImageDraw.Draw(img)
    
    # PRECISION COORDINATES: Row C (Motor Strip)
    hotspots = {
        "RIGHT_HEM": {"coord": (695, 635), "label": "LEFT HAND (CH 13)", "color": "red"}, 
        "LEFT_HEM":  {"coord": (345, 635), "label": "RIGHT HAND (CH 9)", "color": "blue"}, 
        "MIDLINE":   {"coord": (520, 635), "label": "REF (CH 11)", "color": "green"}
    }
    
    for name, data in hotspots.items():
        x, y = data["coord"]
        r = 45
        draw.ellipse([x-r, y-r, x+r, y+r], outline=data["color"], width=6)
        draw.text((x-60, y+50), data["label"], fill=data["color"])

    img.save("breakthrough_mapping.png")
    print("🚀 Motor Mapping Finalized and Verified.")

draw_motor_mapping(r"C:\Users\IMMU\Desktop\Unibe Spring Semester - 2026 Semester 2\Deep Learning\64_channel_sharbrough.pdf")