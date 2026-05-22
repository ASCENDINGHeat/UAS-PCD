1. Open **PowerShell** and install `uv` on Windows:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

```


2. Unzip the file, open PowerShell inside the project folder, and create the Windows virtual environment:
```powershell
uv venv --python 3.14

```


3. Activate the virtual environment on Windows:
```powershell
.venv\Scripts\activate

```


4. Install all the dependencies from your text file:
```powershell
uv pip install -r requirements.txt

```


5. Launch the Django application:
```powershell
python manage.py runserver 0.0.0.0:8000

```
---

## Windows-Specific OpenCV Caveats

The pure Python Django code and HTML will run identically on Windows. However, OpenCV interacts with the Windows camera framework differently than it does with the Linux kernel (`V4L2`).

If encounters a black screen or camera initialization lag, modify the camera initialization line in `monitor/views.py` to explicitly use the Windows DirectShow backend:

```python
# Inside monitor/views.py -> AsynchronousCrackTracker.__init__
# Change this:
self.cap = cv2.VideoCapture(camera_index)

# To this (if Windows struggles tao open the camera framework natively):
self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

```