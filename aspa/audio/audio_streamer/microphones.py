from pydantic import BaseModel


class LocalMicrophone(BaseModel):
    """Local Microphone class."""

    name: str
    index: int
    hostApi: int
    maxInputChannels: int
    maxOutputChannels: int
    defaultLowInputLatency: float
    defaultLowOutputLatency: float
    defaultHighInputLatency: float
    defaultHighOutputLatency: float
    defaultSampleRate: float


class RemoteMicrophone(BaseModel):
    """Remote microphone class."""

    name: str
    ip: str
    port: int
    path: str
    username: str | None = None
    password: str | None = None

    @property
    def rtsp_url(self) -> str:
        return RemoteMicrophone.get_rtsp_url(
            ip=self.ip,
            port=self.port,
            path=self.path,
            username=self.username,
            password=self.password,
        )

    @staticmethod
    def get_rtsp_url(
        ip: str,
        port: int,
        path: str,
        username: str | None = None,
        password: str | None = None,
    ) -> str:
        """Construct an RTSP URL from the given parameters."""
        url: str = f"rtsp://{ip}:{port}/{path}"

        if username and password:
            url = f"rtsp://{username}:{password}@{ip}:{port}/{path}"

        return url
