import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "FoxMesh",
  description: "Blender 기반 지능형 3D 생성 플랫폼",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ko">
      <body>
        <nav className="nav">
          <Link href="/" className="brand">
            Fox<span>Mesh</span>
          </Link>
          <Link href="/" className="link">대시보드</Link>
          <Link href="/generate" className="link">새 3D 모델 생성</Link>
          <Link href="/engines" className="link">실행 엔진 관리</Link>
        </nav>
        <main>{children}</main>
      </body>
    </html>
  );
}
