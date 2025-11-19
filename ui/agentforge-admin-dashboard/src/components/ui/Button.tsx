'use client';

import { forwardRef } from 'react';
import { motion } from 'framer-motion';
import { Loader2 } from 'lucide-react';
import { clsx } from 'clsx';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'default' | 'primary' | 'secondary' | 'danger' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  icon?: React.ReactNode;
}

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'default', size = 'md', loading, icon, children, disabled, ...props }, ref) => {
    const baseClasses = 'inline-flex items-center justify-center gap-2 rounded-xl font-medium transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none';
    
    const variants = {
      default: 'border border-white/10 bg-white/5 hover:bg-white/10 text-day-text dark:text-night-text focus:ring-day-accent/50 dark:focus:ring-night-text/50',
      primary: 'bg-day-accent hover:bg-day-accent/80 text-white border border-day-accent focus:ring-day-accent/50',
      secondary: 'bg-day-lines hover:bg-day-lines/80 text-day-text border border-day-lines focus:ring-day-lines/50',
      danger: 'bg-red-600 hover:bg-red-700 text-white border border-red-600 focus:ring-red-500/50',
      ghost: 'hover:bg-white/5 text-day-text dark:text-night-text focus:ring-day-accent/50 dark:focus:ring-night-text/50'
    };

    const sizes = {
      sm: 'px-3 py-1.5 text-sm',
      md: 'px-4 py-2 text-sm',
      lg: 'px-6 py-3 text-base'
    };

    return (
      <motion.button
        ref={ref}
        className={clsx(baseClasses, variants[variant], sizes[size], className)}
        disabled={disabled || loading}
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        {...(props as any)}
      >
        {loading ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : icon ? (
          icon
        ) : null}
        {children}
      </motion.button>
    );
  }
);

Button.displayName = 'Button';

export { Button };
