# Frontend - Interface Utilisateur

Interface web pour upscaler des images facilement.

## Options de Stack

### Option 1 : MVP Rapide (Recommandé pour démarrer)
- **Framework** : Streamlit ou Gradio
- **Déploiement** : Streamlit Cloud / HuggingFace Spaces
- **Temps de dev** : 1-2 jours
- **Avantages** : Zero frontend code, focus sur la logique

### Option 2 : Production Ready
- **Framework** : Next.js 14 (App Router)
- **Styling** : Tailwind CSS + shadcn/ui
- **State** : Zustand ou React Context
- **Upload** : react-dropzone
- **Déploiement** : Vercel
- **Temps de dev** : 1-2 semaines

## Features UI

### Core Features
- ✅ Upload image (drag & drop)
- ✅ Preview avant/après avec slider
- ✅ Choix du scale (x2, x4)
- ✅ Download résultat
- ✅ Historique des conversions

### Premium Features (optionnel)
- 🔒 Authentication (Clerk/Supabase)
- 🔒 Batch processing
- 🔒 API access
- 🔒 Custom models
- 🔒 Priority queue

## Structure (Next.js)

```
frontend/
├── app/
│   ├── page.tsx              # Homepage
│   ├── upload/
│   │   └── page.tsx          # Upload interface
│   ├── result/
│   │   └── [id]/page.tsx     # Résultat
│   └── api/
│       └── proxy/            # Proxy vers backend
├── components/
│   ├── ui/                   # shadcn components
│   ├── ImageUploader.tsx
│   ├── ImageComparison.tsx   # Slider avant/après
│   └── ProcessingStatus.tsx
├── lib/
│   ├── api.ts                # API client
│   └── utils.ts
├── public/
└── package.json
```

## Wireframe

```
┌─────────────────────────────────────┐
│  🎨 Super Resolution AI             │
├─────────────────────────────────────┤
│                                     │
│  ┌───────────────────────────────┐ │
│  │   Drag & Drop Image Here      │ │
│  │   or Click to Upload          │ │
│  │                               │ │
│  │   📁  Supported: PNG, JPG     │ │
│  └───────────────────────────────┘ │
│                                     │
│  Scale:  ○ 2x   ● 4x   ○ 8x       │
│                                     │
│  [ Upscale Image ]                 │
│                                     │
├─────────────────────────────────────┤
│  Recent Uploads                     │
│  • image1.png  [Download]          │
│  • image2.jpg  [Download]          │
└─────────────────────────────────────┘
```

## TODO

- [ ] Choisir stack (Streamlit vs Next.js)
- [ ] Setup projet
- [ ] Composant upload
- [ ] Composant comparison slider
- [ ] Intégration API backend
- [ ] Responsive design
- [ ] Déploiement

## Démarrage (à venir)

### Streamlit
```bash
cd frontend
pip install streamlit
streamlit run app.py
```

### Next.js
```bash
cd frontend
npm install
npm run dev
```
