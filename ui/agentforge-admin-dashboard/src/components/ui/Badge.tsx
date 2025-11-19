'use client';

import { forwardRef } from 'react';
import { clsx } from 'clsx';

interface BadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'success' | 'warning' | 'danger' | 'info';
  size?: 'sm' | 'md' | 'lg';
}

const Badge = forwardRef<HTMLDivElement, BadgeProps>(
  ({ className, variant = 'default', size = 'md', ...props }, ref) => {
    const baseClasses = 'inline-flex items-center rounded-full border font-medium transition-colors';
    
    const variants = {
      default: 'border-white/20 bg-white/10 text-day-text dark:text-night-text',
      success: 'border-green-500/20 bg-green-500/10 text-green-400',
      warning: 'border-yellow-500/20 bg-yellow-500/10 text-yellow-400',
      danger: 'border-red-500/20 bg-red-500/10 text-red-400',
      info: 'border-day-accent/20 bg-day-accent/10 text-day-accent dark:border-night-text/20 dark:bg-night-text/10 dark:text-night-text'
    };

    const sizes = {
      sm: 'px-2 py-0.5 text-xs',
      md: 'px-2.5 py-0.5 text-xs',
      lg: 'px-3 py-1 text-sm'
    };

    return (
      <div
        ref={ref}
        className={clsx(baseClasses, variants[variant], sizes[size], className)}
        {...props}
      />
    );
  }
);

Badge.displayName = 'Badge';

export { Badge };

