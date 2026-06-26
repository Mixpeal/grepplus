class Grepplus < Formula
  desc "Hybrid code search CLI — grep-fast by default, semantic when you need it"
  homepage "https://github.com/Mixpeal/grepplus"
  url "https://github.com/Mixpeal/grepplus/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "eba656de9b627bf4ed4601f829b70b64b3d45f2e0308d9969fd26131de0b96c1"
  license "Apache-2.0"
  head "https://github.com/Mixpeal/grepplus.git", branch: "main"

  depends_on "rust" => :build

  def install
    system "cargo", "install", *std_cargo_args(path: "crates/gp-cli")
  end

  test do
    assert_match "grepplus", shell_output("#{bin}/grepplus --help")
    assert_path_exists bin/"gp"
  end
end
