import argparse
import ollama
from ollama import Client
from PIL import Image
from io import BytesIO

def process_image(image_file, host, port, prompt):
    client = Client(host=f"http://{host}:{port}")
    
    with Image.open(image_file) as img:
        with BytesIO() as buffer:
            img.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()

            full_response = ''
            
            for response in client.generate(
                model='llava:13b',
                prompt=prompt,
                images=[image_bytes],
                stream=True
            ):
                # print(response['response'], end='', flush=True)
                full_response += response['response']

            return full_response

def main():
    parser = argparse.ArgumentParser(description="Process an image using Ollama's LLaVA model.")
    parser.add_argument("image", help="Path to the image file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode")
    parser.add_argument("--host", default="localhost", help="Host address of the Ollama server")
    parser.add_argument("--port", default=11434, help="Port number of the Ollama server")
    parser.add_argument("--prompt", default="describe the image", help="Prompt to guide the model")
    args = parser.parse_args()
    
    response = process_image(args.image, args.host, args.port, args.prompt)
    print(response)

if __name__ == '__main__':
    main()
