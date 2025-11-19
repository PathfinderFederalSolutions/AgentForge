'use client';

import { Bell, Search, User, Settings, LogOut } from 'lucide-react';
import { ThemeToggle } from '../ThemeToggle';
import { Input } from '../ui/Input';
import { Button } from '../ui/Button';
import { Badge } from '../ui/Badge';
import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useSnapshot } from 'valtio';
import { store } from '@/lib/state';

export function Header() {
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [showNotifications, setShowNotifications] = useState(false);
  const snap = useSnapshot(store);

  const notifications = [
    { id: 1, title: 'Agent Deployment Complete', message: 'GPU cluster successfully deployed', time: '2 min ago', type: 'success' },
    { id: 2, title: 'High Memory Usage', message: 'Neural mesh approaching capacity limits', time: '5 min ago', type: 'warning' },
    { id: 3, title: 'Job Queue Alert', message: '50+ jobs pending in quantum scheduler', time: '10 min ago', type: 'info' },
  ];

  return (
    <header className="fixed top-0 right-0 left-0 lg:left-80 h-16 z-30 hud-card border-b border-white/10 dark:border-red-900/40">
      <div className="flex items-center justify-between h-full px-6">
        {/* Search */}
        <div className="flex-1 max-w-md">
          <Input
            variant="search"
            placeholder="Search agents, jobs, or commands..."
            className="w-full"
          />
        </div>

        {/* Actions */}
        <div className="flex items-center gap-4">
          {/* Connection Status */}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5 dark:bg-black/20">
            <div className={`w-2 h-2 rounded-full ${snap.connected ? 'bg-green-400' : 'bg-red-400'} animate-pulse`} />
            <span className="text-xs font-medium">
              {snap.connected ? 'Connected' : 'Disconnected'}
            </span>
          </div>

          {/* Notifications */}
          <div className="relative">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowNotifications(!showNotifications)}
              icon={<Bell className="h-4 w-4" />}
            />
            {notifications.length > 0 && (
              <Badge
                variant="danger"
                size="sm"
                className="absolute -top-1 -right-1 px-1.5 min-w-[1.25rem] h-5 text-xs"
              >
                {notifications.length}
              </Badge>
            )}

            {/* Notifications Dropdown */}
            <AnimatePresence>
              {showNotifications && (
                <motion.div
                  initial={{ opacity: 0, y: -10, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: -10, scale: 0.95 }}
                  className="absolute right-0 mt-2 w-80 hud-card border border-white/10 dark:border-red-900/40 p-4 space-y-3"
                >
                  <div className="flex items-center justify-between">
                    <h3 className="label">Notifications</h3>
                    <Button variant="ghost" size="sm" onClick={() => setShowNotifications(false)}>
                      Clear All
                    </Button>
                  </div>
                  
                  <div className="space-y-2 max-h-64 overflow-y-auto">
                    {notifications.map((notification) => (
                      <div
                        key={notification.id}
                        className="p-3 rounded-lg bg-white/5 dark:bg-black/20 border border-white/10 dark:border-red-900/40"
                      >
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <div className="flex items-center gap-2">
                              <h4 className="text-sm font-medium">{notification.title}</h4>
                              <Badge 
                                variant={notification.type as any} 
                                size="sm"
                              >
                                {notification.type}
                              </Badge>
                            </div>
                            <p className="text-xs opacity-70 mt-1">{notification.message}</p>
                            <p className="text-xs opacity-50 mt-2">{notification.time}</p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          <ThemeToggle />

          {/* User Menu */}
          <div className="relative">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowUserMenu(!showUserMenu)}
              icon={<User className="h-4 w-4" />}
            />

            {/* User Dropdown */}
            <AnimatePresence>
              {showUserMenu && (
                <motion.div
                  initial={{ opacity: 0, y: -10, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: -10, scale: 0.95 }}
                  className="absolute right-0 mt-2 w-48 hud-card border border-white/10 dark:border-red-900/40 p-2"
                >
                  <div className="px-3 py-2 border-b border-white/10 dark:border-red-900/40 mb-2">
                    <div className="text-sm font-medium">Admin User</div>
                    <div className="text-xs opacity-60">admin@agentforge.ai</div>
                  </div>
                  
                  <div className="space-y-1">
                    <Button
                      variant="ghost"
                      size="sm"
                      className="w-full justify-start"
                      icon={<Settings className="h-4 w-4" />}
                    >
                      Settings
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="w-full justify-start text-red-400 hover:text-red-300"
                      icon={<LogOut className="h-4 w-4" />}
                    >
                      Sign Out
                    </Button>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </header>
  );
}
