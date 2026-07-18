# System Operations Manual — QC-Vision Bottle Detection

> A complete **Standard Operating Procedure (SOP)** for daily operation,
> monitoring, admin configuration, and troubleshooting.

---

## 1. System Startup (Daily Operation)

This is how you turn the factory line **ON** at the start of a shift.

| Step | Command / Action |
|------|-----------------|
| 1 | Power on the Raspberry Pi |
| 2 | SSH in from laptop: `ssh project_2026@<PI_IP_ADDRESS>` |
| 3 | Navigate to project: `cd ~/bottle_project` |
| 4 | Launch in detached mode: `docker compose up -d` |

The system will boot, load the TFLite model, and start the camera automatically.

---

## 2. System Monitoring (The Dashboard)

1. Open a browser at `http://<PI_IP>:8000`
2. Monitor the **Real-Time Dashboard**
3. When a defect is found:
   - The box turns **RED** (Visual Alert)
   - A "Defect Detected" entry appears in the log list
   - The event is saved to the SQLite database

---

## 3. Admin Configuration (Super Admin Role)

| Step | Action |
|------|--------|
| 1 | Go to `http://<PI_IP>:8000/admin` |
| 2 | Login as the Superuser |
| 3 | Click **System Configs** |
| 4 | Adjust **Confidence Threshold** (e.g., `0.60` → `0.75`) to reduce false alarms |
| 5 | Upload a new `.tflite` model via the **Active Model** field without touching code |

---

## 4. System Shutdown (End of Shift)

```bash
docker compose down
```

This stops the camera and saves the database state.

---

## 5. Troubleshooting & Maintenance

### A. Check Logs (if it crashes)

```bash
docker compose logs -f qc-vision
# Press CTRL+C to exit
```

### B. Sync Code Changes from Mac → Pi

```bash
rsync -av \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude 'db.sqlite3' \
  core/ project_2026@<PI_IP>:~/bottle_project/core/
```

Then restart the container to pick up changes:

```bash
docker compose restart qc-vision
```

### C. Reset the Database (Nuclear Option)

```bash
docker compose down
rm core/db.sqlite3
docker compose up -d
# Re-run the createsuperuser command
```

---

## 6. Useful Quick Commands

| Purpose | Command |
|---------|---------|
| Kill all running Docker tasks | `pkill -9 -f "docker compose"` |
| Find Pi IP address | `hostname -I` |
| SSH into Pi | `ssh project_2026@iotpi.local` |
| Django shell | `python manage.py shell` |
| Create superuser | `python manage.py createsuperuser` |

---

## 7. Architecture Summary

| Layer | Technology |
|-------|-----------|
| Edge Device | Raspberry Pi 4 Model B |
| Inference | TensorFlow Lite (`.tflite`) |
| Web Backend | Django 4.2 |
| Database | SQLite |
| Camera | USB Webcam via V4L2 + MJPEG |
| Deployment | Docker Compose |
