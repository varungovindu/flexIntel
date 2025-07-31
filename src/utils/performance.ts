// Performance monitoring and optimization utilities

export interface PerformanceMetrics {
  loadTime: number;
  renderTime: number;
  memoryUsage: number;
  fps: number;
}

class PerformanceMonitor {
  private metrics: PerformanceMetrics[] = [];
  private startTime: number = 0;
  private frameCount: number = 0;
  private lastFrameTime: number = 0;

  startMonitoring() {
    this.startTime = performance.now();
    this.frameCount = 0;
    this.lastFrameTime = performance.now();
    
    // Monitor FPS
    this.monitorFPS();
    
    // Monitor memory usage
    this.monitorMemory();
  }

  private monitorFPS() {
    const measureFPS = () => {
      const currentTime = performance.now();
      this.frameCount++;
      
      if (currentTime - this.lastFrameTime >= 1000) {
        const fps = Math.round((this.frameCount * 1000) / (currentTime - this.lastFrameTime));
        this.updateMetrics({ fps });
        
        this.frameCount = 0;
        this.lastFrameTime = currentTime;
      }
      
      requestAnimationFrame(measureFPS);
    };
    
    requestAnimationFrame(measureFPS);
  }

  private monitorMemory() {
    if ('memory' in performance) {
      setInterval(() => {
        const memory = (performance as any).memory;
        const memoryUsage = memory.usedJSHeapSize / memory.jsHeapSizeLimit * 100;
        this.updateMetrics({ memoryUsage });
      }, 5000);
    }
  }

  private updateMetrics(partial: Partial<PerformanceMetrics>) {
    const currentMetrics = this.metrics[this.metrics.length - 1] || {
      loadTime: 0,
      renderTime: 0,
      memoryUsage: 0,
      fps: 0
    };
    
    this.metrics.push({
      ...currentMetrics,
      ...partial
    });
  }

  endMonitoring() {
    const endTime = performance.now();
    const loadTime = endTime - this.startTime;
    
    this.updateMetrics({ loadTime });
    
    // Log performance data
    console.log('Performance Metrics:', this.metrics[this.metrics.length - 1]);
    
    return this.metrics[this.metrics.length - 1];
  }

  getMetrics(): PerformanceMetrics[] {
    return [...this.metrics];
  }
}

// Debounce utility for performance optimization
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout;
  
  return (...args: Parameters<T>) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
}

// Throttle utility for performance optimization
export function throttle<T extends (...args: any[]) => any>(
  func: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle: boolean;
  
  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      func(...args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  };
}

// Memoization utility for expensive calculations
export function memoize<T extends (...args: any[]) => any>(
  func: T,
  resolver?: (...args: Parameters<T>) => string
): T {
  const cache = new Map<string, ReturnType<T>>();
  
  return ((...args: Parameters<T>) => {
    const key = resolver ? resolver(...args) : JSON.stringify(args);
    
    if (cache.has(key)) {
      return cache.get(key);
    }
    
    const result = func(...args);
    cache.set(key, result);
    
    return result;
  }) as T;
}

// Image optimization utility
export function preloadImage(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve();
    img.onerror = reject;
    img.src = src;
  });
}

// Bundle size optimization utility
export function lazyLoad<T>(
  importFunc: () => Promise<{ default: T }>,
  fallback?: React.ComponentType
): React.LazyExoticComponent<React.ComponentType<T>> {
  return React.lazy(() => 
    importFunc().catch(() => {
      if (fallback) {
        return { default: fallback };
      }
      throw new Error('Failed to load component');
    })
  );
}

// Performance observer for monitoring
export function observePerformance() {
  if ('PerformanceObserver' in window) {
    const observer = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (entry.entryType === 'measure') {
          console.log(`${entry.name}: ${entry.duration}ms`);
        }
      }
    });
    
    observer.observe({ entryTypes: ['measure'] });
  }
}

// Memory cleanup utility
export function cleanupMemory() {
  if ('gc' in window) {
    (window as any).gc();
  }
  
  // Clear any cached data
  if ('caches' in window) {
    caches.keys().then(names => {
      names.forEach(name => caches.delete(name));
    });
  }
}

export const performanceMonitor = new PerformanceMonitor();