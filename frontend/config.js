// Runtime configuration — edit these two values per environment, or
// override them at build time via your host's env-var injection.
//
// NOTE: This API key is a lightweight abuse-prevention gate, not a
// secret — anything shipped to a browser is visible to the client.
// For real user-level security, put a proper auth layer in front of
// the backend instead of relying on this key alone.
window.APP_CONFIG = {
  BACKEND_URL: "http://localhost:8000",
  API_KEY: "change_me_to_a_long_random_string",
};
