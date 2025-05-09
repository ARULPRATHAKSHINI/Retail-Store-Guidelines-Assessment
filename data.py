from simple_image_download import simple_image_download as simp

# Initialize downloader
response = simp.simple_image_download()

# Download compliant store images (clean/organized)
response.download("well organized retail store interior", limit=50)
print("Downloaded compliant store images")

# Download non-compliant images (cluttered/messy)
response.download("cluttered retail store interior", limit=50)
print("Downloaded non-compliant store images")

# This will automatically create:
# ./simple_images/well organized retail store interior/
# ./simple_images/cluttered retail store interior/