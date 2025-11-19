'use client';

import { forwardRef } from 'react';
import { clsx } from 'clsx';
import { Search, Eye, EyeOff } from 'lucide-react';
import { useState } from 'react';

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  icon?: React.ReactNode;
  variant?: 'default' | 'search';
}

const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, label, error, icon, variant = 'default', ...props }, ref) => {
    const [showPassword, setShowPassword] = useState(false);
    
    const inputType = type === 'password' && showPassword ? 'text' : type;
    
    return (
      <div className="w-full">
        {label && (
          <label className="label block mb-2">
            {label}
          </label>
        )}
        <div className="relative">
          {(icon || variant === 'search') && (
            <div className="absolute left-3 top-1/2 transform -translate-y-1/2 text-day-text/50 dark:text-night-text/50">
              {variant === 'search' ? <Search className="h-4 w-4" /> : icon}
            </div>
          )}
          <input
            type={inputType}
            className={clsx(
              'input',
              (icon || variant === 'search') && 'pl-10',
              type === 'password' && 'pr-10',
              error && 'border-red-500 focus:border-red-500',
              className
            )}
            ref={ref}
            {...props}
          />
          {type === 'password' && (
            <button
              type="button"
              className="absolute right-3 top-1/2 transform -translate-y-1/2 text-day-text/50 dark:text-night-text/50 hover:text-day-text dark:hover:text-night-text transition-colors"
              onClick={() => setShowPassword(!showPassword)}
            >
              {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
            </button>
          )}
        </div>
        {error && (
          <p className="mt-1 text-sm text-red-500">{error}</p>
        )}
      </div>
    );
  }
);

Input.displayName = 'Input';

export { Input };

