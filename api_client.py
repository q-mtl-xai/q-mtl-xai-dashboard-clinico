
import requests

class APIClient:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip("/")
        self.token = None

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def health(self):
        return requests.get(f"{self.base_url}/health", timeout=120).json()

    def login(self, email, password):
        r = requests.post(f"{self.base_url}/auth/login",
                          json={"email": email, "password": password}, timeout=120)
        r.raise_for_status()
        data = r.json()
        self.token = data["access_token"]
        return data

    def register(self, email, password, full_name, medical_license=None):
        r = requests.post(f"{self.base_url}/auth/register",
                          json={"email": email, "password": password,
                                "full_name": full_name, "medical_license": medical_license},
                          timeout=120)
        r.raise_for_status()
        data = r.json()
        self.token = data["access_token"]
        return data

    def search_patient(self, dni):
        r = requests.get(f"{self.base_url}/patients/by-dni/{dni}",
                         headers=self._headers(), timeout=120)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

    def create_patient(self, dni, full_name, birth_date=None, gender=None):
        r = requests.post(f"{self.base_url}/patients",
                          json={"dni": dni, "full_name": full_name,
                                "birth_date": birth_date, "gender": gender},
                          headers=self._headers(), timeout=120)
        r.raise_for_status()
        return r.json()

    def list_studies(self, patient_id):
        r = requests.get(f"{self.base_url}/patients/{patient_id}/studies",
                         headers=self._headers(), timeout=120)
        r.raise_for_status()
        return r.json()["studies"]

    def get_study(self, study_id):
        r = requests.get(f"{self.base_url}/studies/{study_id}",
                         headers=self._headers(), timeout=120)
        r.raise_for_status()
        return r.json()

    # === NUEVO MÉTODO PARA DESCARGAR LA SEÑAL ===
    def get_study_signal(self, study_id):
        r = requests.get(f"{self.base_url}/studies/{study_id}/signal",
                         headers=self._headers(), timeout=120)
        r.raise_for_status()
        return r.json()["signal"]

    def predict_only(self, signal, task="isc", sampling_rate=250):
        r = requests.post(f"{self.base_url}/predict",
                          json={"signal": signal.flatten().tolist() if hasattr(signal, "flatten") else signal,
                                "task": task, "sampling_rate": sampling_rate},
                          headers=self._headers(), timeout=180)
        r.raise_for_status()
        return r.json()

    def explain(self, prediction, confidence, sqi, top_timesteps, task="isc"):
        r = requests.post(f"{self.base_url}/studies/explain",
                          json={"prediction": prediction, "confidence": confidence,
                                "sqi": sqi, "top_timesteps": top_timesteps, "task": task},
                          headers=self._headers(), timeout=180)
        r.raise_for_status()
        return r.json()

    def explain_saved(self, study_id):
        r = requests.post(f"{self.base_url}/studies/{study_id}/explain",
                          headers=self._headers(), timeout=180)
        r.raise_for_status()
        return r.json()

    def save_analyzed_study(self, patient_id, signal, task, medical_validation, medical_notes):
        r = requests.post(f"{self.base_url}/studies/analyze",
                          json={"patient_id": patient_id,
                                "signal": signal.flatten().tolist() if hasattr(signal, "flatten") else signal,
                                "task": task, "sampling_rate": 250,
                                "medical_validation": medical_validation,
                                "medical_notes": medical_notes},
                          headers=self._headers(), timeout=180)
        r.raise_for_status()
        return r.json()

    def update_validation(self, study_id, medical_validation, medical_notes):
        r = requests.patch(f"{self.base_url}/studies/{study_id}",
                           json={"medical_validation": medical_validation, "medical_notes": medical_notes},
                           headers=self._headers(), timeout=120)
        r.raise_for_status()
        return r.json()

    def void_study(self, study_id, void_reason):
        r = requests.patch(f"{self.base_url}/studies/{study_id}/void",
                           json={"void_reason": void_reason},
                           headers=self._headers(), timeout=120)
        r.raise_for_status()
        return r.json()
