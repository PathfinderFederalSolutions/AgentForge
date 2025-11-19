'use client';

import { forwardRef } from 'react';
import { motion } from 'framer-motion';
import { clsx } from 'clsx';

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'glass' | 'solid';
  hover?: boolean;
}

const Card = forwardRef<HTMLDivElement, CardProps>(
  ({ className, variant = 'default', hover = false, children, ...props }, ref) => {
    const baseClasses = 'rounded-2xl backdrop-blur border transition-all duration-200';
    
    const variants = {
      default: 'hud-card',
      glass: 'bg-white/5 dark:bg-black/30 border-white/10 dark:border-red-900/40 shadow-hud',
      solid: 'bg-day-grid dark:bg-night-grid border-day-lines dark:border-red-900/40'
    };

    const Component = hover ? motion.div : 'div';
    const motionProps = hover ? {
      whileHover: { scale: 1.02, y: -2 },
      transition: { type: 'spring', stiffness: 300, damping: 20 }
    } : {};

    return (
      <Component
        ref={ref}
        className={clsx(baseClasses, variants[variant], className)}
        {...motionProps}
        {...(props as any)}
      >
        {children}
      </Component>
    );
  }
);

Card.displayName = 'Card';

const CardHeader = forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={clsx('flex flex-col space-y-1.5 p-6', className)} {...props} />
  )
);

CardHeader.displayName = 'CardHeader';

const CardTitle = forwardRef<HTMLParagraphElement, React.HTMLAttributes<HTMLHeadingElement>>(
  ({ className, ...props }, ref) => (
    <h3 ref={ref} className={clsx('label text-sm font-medium leading-none tracking-tight', className)} {...props} />
  )
);

CardTitle.displayName = 'CardTitle';

const CardDescription = forwardRef<HTMLParagraphElement, React.HTMLAttributes<HTMLParagraphElement>>(
  ({ className, ...props }, ref) => (
    <p ref={ref} className={clsx('text-sm text-day-text/70 dark:text-night-text/70', className)} {...props} />
  )
);

CardDescription.displayName = 'CardDescription';

const CardContent = forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={clsx('p-6 pt-0', className)} {...props} />
  )
);

CardContent.displayName = 'CardContent';

const CardFooter = forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={clsx('flex items-center p-6 pt-0', className)} {...props} />
  )
);

CardFooter.displayName = 'CardFooter';

export { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter };
