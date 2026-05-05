# InSpec AI Frontend

Next.js frontend for the InSpec AI dashboard.

## Run

From this folder:

```bat
npm install
npm run dev
```

Open:

```text
http://localhost:3000
```

The frontend uses the unified backend at:

```text
http://localhost:8000
```

Image inference uses `/api/predict`; video inference uses
`/api/video/predict`.
