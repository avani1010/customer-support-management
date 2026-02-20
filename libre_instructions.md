# LibreTranslate (self-hosted) + German→English demo
Basically LibreTranslate is open source translation tool, so you can download once and then run it on your own docker server. 

## Prerequisites
- Docker Desktop installed and running.

## 1) Start LibreTranslate (DE+EN only)
We map container port `5000` to host port `5001` to avoid conflicts on port 5000.

> Note: LibreTranslate defaults to `--host 127.0.0.1` and `--port 5000`.
> For Docker port publishing, we bind to all interfaces: `--host 0.0.0.0`.

### Run (foreground)
```bash
docker run --rm -it \
  -p 5001:5000 \
  -e LT_LOAD_ONLY=de,en \
  libretranslate/libretranslate \
  --host 0.0.0.0 --port 5000
```
### Keep this terminal open. Stopping it stops the API (because of --rm).

### 2) Verify the API is working
```bash
curl -4 -s http://127.0.0.1:5001/languages | head
```
### You should see JSON with en and de like this.
### [{"code":"en","name":"English","targets":["de","en"]},{"code":"de","name":"German","targets":["de","en"]}]