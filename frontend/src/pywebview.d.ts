interface Window {
  pywebview?: {
    api?: {
      switchToManagedRuntime: () => Promise<void>;
    };
  };
}
