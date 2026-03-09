# AutoQR

Script Python pour analyser une video YouTube ou un fichier video local, detecter les QR codes affiches, sauvegarder leur image, enregistrer leur contenu et leurs timestamps, puis produire un resume terminal.

Le script principal est :

- `scan_youtube_qr.py`

## Ce que fait le script

- recupere les metadonnees et formats via `yt-dlp` pour une URL YouTube
- peut scanner directement un fichier local avec `--video-file`
- detecte plusieurs QR dans une meme frame
- sauvegarde une image PNG par QR detecte
- exporte les resultats en `CSV` et `JSON`
- deduplique les QR repetes
- affiche les liens web trouves sans doublons dans le terminal
- affiche un timer turbo global du debut de la phase 1 jusqu'a la fin

## Modes disponibles

### 1. Fichier local

C'est le mode recommande si tu veux :

- eviter les problemes YouTube/cookies
- rescanner plusieurs fois la meme video
- avoir le workflow le plus stable

Exemple :

```bash
python scan_youtube_qr.py --video-file "/chemin/vers/video.mp4" --turbo-precise --output qr_results
```

### 2. URL YouTube

Le script peut aussi prendre une URL YouTube en entree.

Exemple :

```bash
python scan_youtube_qr.py "https://www.youtube.com/watch?v=XXXXXXXXXXX" --turbo-precise --output qr_results
```

Si YouTube demande une authentification :

```bash
python scan_youtube_qr.py "URL" --cookies "/chemin/vers/cookies.txt" --turbo-precise --output qr_results
```

## Sorties generees

Dans le dossier passe avec `--output`, le script genere :

- un dossier `images` contenant les PNG des QR trouves
- un fichier `qr_results.csv`
- un fichier `qr_results.json`

Chaque resultat contient :

- `timestamp_seconds`
- `timestamp_hhmmss_ms`
- `qr_content`
- `image_file`
- `frame_number`

## Installation

## Windows

### 1. Installer Python

Installe Python 3.12 ou plus recent.

### 2. Installer FFmpeg

```powershell
winget install Gyan.FFmpeg
```

Verifie ensuite :

```powershell
ffmpeg -version
ffprobe -version
```

### 3. Installer les dependances Python

Si tu n'as pas encore de `requirements.txt`, tu peux installer directement :

```powershell
python -m pip install yt-dlp opencv-python numpy zxing-cpp
```

## macOS

### 1. Installer Python et FFmpeg

```bash
brew install python@3.12 ffmpeg
```

### 2. Creer un environnement virtuel

Depuis le dossier du projet :

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install yt-dlp opencv-python numpy zxing-cpp
```

Ensuite, a chaque nouveau terminal :

```bash
cd "/chemin/vers/Autoqr"
source .venv/bin/activate
```

## Utilisation rapide

## Scan local recommande

```powershell
python scan_youtube_qr.py --video-file "C:\chemin\vers\video.mp4" --turbo-precise --output qr_results
```

ou sur macOS :

```bash
python scan_youtube_qr.py --video-file "/chemin/vers/video.mp4" --turbo-precise --output qr_results
```

## Scan YouTube

```powershell
python scan_youtube_qr.py "https://www.youtube.com/watch?v=XXXXXXXXXXX" --turbo-precise --output qr_results
```

## Parametres utiles

### Parametres generaux

- `url` : URL YouTube a scanner
- `--video-file` : chemin d'un fichier local
- `--output` : dossier de sortie
- `--fps-scan` : cadence de scan simple
- `--full-scan` : analyse dense
- `--min-interval` : intervalle minimal pour la deduplication
- `--prefer-download` : prefere le mode fichier local pour une URL YouTube
- `--keep-temp` : conserve la video temporaire en mode download

### Authentification YouTube

- `--cookies`
- `--cookies-from-browser`
- `--js-runtime`
- `--remote-component`
- `--max-height`

### Mode turbo

- `--turbo-precise`
- `--turbo-prescan-fps`
- `--turbo-scale-width`
- `--turbo-window`
- `--turbo-merge-gap`
- `--turbo-motion-threshold`
- `--turbo-max-skip-frames`
- `--turbo-prescan-color`
- `--turbo-workers`
- `--turbo-start-guard`

## Presets recommandes

## 1. Equilibre vitesse / precision

Bon preset pour une video locale avec QR de taille moyenne :

```bash
python scan_youtube_qr.py --video-file "/chemin/vers/video.mp4" --turbo-precise --turbo-prescan-fps 25 --turbo-scale-width 1280 --turbo-max-skip-frames 0 --turbo-motion-threshold 0.0 --turbo-workers 8 --output qr_results
```

## 2. Plus rapide

Si les QR sont assez gros :

```bash
python scan_youtube_qr.py --video-file "/chemin/vers/video.mp4" --turbo-precise --turbo-prescan-fps 25 --turbo-scale-width 854 --turbo-max-skip-frames 0 --turbo-motion-threshold 0.0 --turbo-workers 8 --output qr_results
```

## 3. Securite max

Si tu veux minimiser le risque de rater un petit QR :

```bash
python scan_youtube_qr.py --video-file "/chemin/vers/video.mp4" --turbo-precise --turbo-prescan-fps 25 --turbo-scale-width 1920 --turbo-max-skip-frames 0 --turbo-motion-threshold 0.0 --turbo-workers 8 --output qr_results
```

## A propos du fps

Si ton objectif est de ne pas rater un QR tres bref :

- scanne au `fps` natif de la video
- ne mets pas un `fps` superieur au `fps` source
- desactive le skip si tu veux la securite maximale

Exemples :

- video 25 fps -> `--turbo-prescan-fps 25`
- video 30 fps -> `--turbo-prescan-fps 30`

## A propos de la resolution de la phase 1

Repere rapide :

- 480p -> `--turbo-scale-width 854`
- 720p -> `--turbo-scale-width 1280`
- 1080p -> `--turbo-scale-width 1920`

Plus la valeur est haute :

- plus la phase 1 est lente
- plus elle est robuste sur les petits QR

## Exemples de logs

### Nouveau QR detecte

```text
[QR] 00:12.340 | contenu=https://exemple.com | image=qr_0003_00-12-340.png
```

### Lien unique affiche dans le terminal

```text
[LINK] https://exemple.com
```

### Resume final

```text
=== RESUME ===
Titre: ...
Resolution choisie: ...
Mode utilise: ...
Frames analysees: ...
Turbo pass1 frames: ...
Turbo pass2 frames: ...
Turbo temps total: ...
QR uniques trouves: ...
Liens QR uniques:
  https://exemple.com
CSV: ...
JSON: ...
```

## Conseils pratiques

- privilegie `--video-file` si tu veux un workflow stable
- garde la video locale si tu veux rescanner avec d'autres reglages
- commence en `1280` si tu n'es pas sur de la taille des QR
- descends a `854` seulement si les QR sont clairement gros
- monte a `1920` si tu as le moindre doute sur un QR rate

## Depannage

## `ffmpeg was not found in PATH`

Installe FFmpeg et verifie :

```bash
ffmpeg -version
ffprobe -version
```

## `Sign in to confirm you're not a bot`

Ca vient de YouTube, pas du scan QR.

Solution la plus fiable :

- telecharger la video une fois
- rescanner ensuite uniquement avec `--video-file`

## `externally-managed-environment` sur macOS

Utilise un environnement virtuel :

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install yt-dlp opencv-python numpy zxing-cpp
```

## Le script semble lent

Regarde dans le resume :

- `Turbo pass1 frames`
- `Turbo pass2 frames`
- `Turbo temps total`

Le plus gros levier est souvent :

- le `fps` de prescan
- `--turbo-scale-width`
- la presence ou non de `skip`

## Raccourci macOS optionnel

Tu peux ajouter dans `~/.zshrc` :

```bash
alias autoqr='cd "/Users/joseph/Autoqr" && source "/Users/joseph/Autoqr/.venv/bin/activate"'
```

Puis :

```bash
source ~/.zshrc
autoqr
```

## Fichier principal

- `scan_youtube_qr.py`

## Licence / usage

Assure-toi d'avoir les droits necessaires pour analyser les videos que tu utilises, en particulier si tu travailles a partir d'une URL YouTube.
