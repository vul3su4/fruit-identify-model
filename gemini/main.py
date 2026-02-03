from google import genai
from google.genai import types

#  `GEMINI_API_KEY`.
client = genai.Client(api_key="PUT YOUR API KEY HERE")

data_image_location = "PUT YOUR IMAGE PATH HERE"

with open(data_image_location, "rb") as f:
    image_bytes = f.read()
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=[
        types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/jpeg",
        ),
        "Identify the item in the image",
    ]
)
print(response.text)
