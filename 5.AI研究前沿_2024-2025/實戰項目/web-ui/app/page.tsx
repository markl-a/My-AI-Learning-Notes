'use client'

import { useState } from 'react'
import Link from 'next/link'
import {
  MessageSquare,
  FileSearch,
  Code,
  Shield,
  Zap,
  ArrowRight
} from 'lucide-react'

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <h1 className="text-2xl font-bold text-gray-900">
                AI Assistant Platform
              </h1>
            </div>
            <nav className="hidden md:flex space-x-8">
              <a href="#features" className="text-gray-700 hover:text-blue-600">
                功能
              </a>
              <a href="#projects" className="text-gray-700 hover:text-blue-600">
                項目
              </a>
              <a href="#docs" className="text-gray-700 hover:text-blue-600">
                文檔
              </a>
            </nav>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="text-center">
          <h2 className="text-5xl font-extrabold text-gray-900 sm:text-6xl">
            智能 AI 助手平台
          </h2>
          <p className="mt-6 text-xl text-gray-600 max-w-3xl mx-auto">
            集成多個 AI 驅動的工具，幫助您提高生產力、代碼質量和文檔管理
          </p>
          <div className="mt-10 flex justify-center gap-4">
            <Link
              href="/chat"
              className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700"
            >
              開始使用
              <ArrowRight className="ml-2 h-5 w-5" />
            </Link>
            <a
              href="#projects"
              className="inline-flex items-center px-6 py-3 border border-gray-300 text-base font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50"
            >
              查看項目
            </a>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="text-center mb-12">
          <h3 className="text-3xl font-bold text-gray-900">核心功能</h3>
          <p className="mt-4 text-lg text-gray-600">
            強大的 AI 功能，助力您的工作流程
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <FeatureCard
            icon={<MessageSquare className="h-8 w-8 text-blue-600" />}
            title="智能對話"
            description="基於 RAG 技術的智能問答系統，理解上下文並提供準確答案"
          />
          <FeatureCard
            icon={<FileSearch className="h-8 w-8 text-blue-600" />}
            title="文檔分析"
            description="自動分析文檔內容，提取關鍵信息、生成摘要和問答"
          />
          <FeatureCard
            icon={<Code className="h-8 w-8 text-blue-600" />}
            title="代碼審查"
            description="AI 驅動的代碼審查，識別問題並提供優化建議"
          />
          <FeatureCard
            icon={<Shield className="h-8 w-8 text-blue-600" />}
            title="安全檢測"
            description="自動掃描代碼漏洞，符合 OWASP Top 10 標準"
          />
          <FeatureCard
            icon={<Zap className="h-8 w-8 text-blue-600" />}
            title="性能分析"
            description="分析代碼性能瓶頸，提供優化方案"
          />
          <FeatureCard
            icon={<MessageSquare className="h-8 w-8 text-blue-600" />}
            title="自動化測試"
            description="自動生成單元測試和文檔"
          />
        </div>
      </section>

      {/* Projects Section */}
      <section id="projects" className="bg-white py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h3 className="text-3xl font-bold text-gray-900">可用項目</h3>
            <p className="mt-4 text-lg text-gray-600">
              選擇一個項目開始使用
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <ProjectCard
              title="RAG ChatBot"
              description="基於檢索增強生成的智能聊天機器人"
              href="/chat"
              features={[
                '文檔檢索',
                '上下文對話',
                '多輪對話記憶'
              ]}
            />
            <ProjectCard
              title="文檔分析系統"
              description="智能文檔處理和分析平台"
              href="/documents"
              features={[
                '多格式支持',
                '智能摘要',
                '實體提取'
              ]}
            />
            <ProjectCard
              title="代碼審查助手"
              description="AI 驅動的代碼質量分析工具"
              href="/code-review"
              features={[
                '安全掃描',
                '性能分析',
                '最佳實踐檢查'
              ]}
            />
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-800 text-white py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div>
              <h4 className="text-lg font-semibold mb-4">AI Assistant Platform</h4>
              <p className="text-gray-400">
                打造智能化的工作流程
              </p>
            </div>
            <div>
              <h4 className="text-lg font-semibold mb-4">快速鏈接</h4>
              <ul className="space-y-2">
                <li><a href="/docs" className="text-gray-400 hover:text-white">文檔</a></li>
                <li><a href="/api" className="text-gray-400 hover:text-white">API</a></li>
                <li><a href="/blog" className="text-gray-400 hover:text-white">博客</a></li>
              </ul>
            </div>
            <div>
              <h4 className="text-lg font-semibold mb-4">聯繫我們</h4>
              <p className="text-gray-400">
                有問題？提交 Issue 或聯繫我們
              </p>
            </div>
          </div>
          <div className="mt-8 pt-8 border-t border-gray-700 text-center text-gray-400">
            <p>&copy; 2024 AI Assistant Platform. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  )
}

function FeatureCard({ icon, title, description }: {
  icon: React.ReactNode
  title: string
  description: string
}) {
  return (
    <div className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow">
      <div className="flex items-center justify-center h-12 w-12 rounded-md bg-blue-100 mb-4">
        {icon}
      </div>
      <h4 className="text-lg font-semibold text-gray-900 mb-2">{title}</h4>
      <p className="text-gray-600">{description}</p>
    </div>
  )
}

function ProjectCard({ title, description, href, features }: {
  title: string
  description: string
  href: string
  features: string[]
}) {
  return (
    <div className="bg-white border border-gray-200 rounded-lg shadow-sm hover:shadow-md transition-shadow p-6">
      <h4 className="text-xl font-bold text-gray-900 mb-2">{title}</h4>
      <p className="text-gray-600 mb-4">{description}</p>
      <ul className="space-y-2 mb-6">
        {features.map((feature, index) => (
          <li key={index} className="flex items-center text-sm text-gray-600">
            <svg className="h-4 w-4 text-green-500 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
            {feature}
          </li>
        ))}
      </ul>
      <Link
        href={href}
        className="inline-flex items-center justify-center w-full px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700"
      >
        開始使用
        <ArrowRight className="ml-2 h-4 w-4" />
      </Link>
    </div>
  )
}
