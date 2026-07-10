class Grepplus < Formula
  desc "Hybrid code search CLI — grep-fast by default, semantic when you need it"
  homepage "https://github.com/Mixpeal/grepplus"
  url "https://github.com/Mixpeal/grepplus/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "d2d4b0932fc782dab0b205d44b2ff83cfde71c4fe778c589469fc6e5d0d7440d"
  license "Apache-2.0"
  head "https://github.com/Mixpeal/grepplus.git", branch: "main"

  depends_on "rust" => :build

  def install
    system "cargo", "install", *std_cargo_args(path: ".")
  end

  test do
    assert_match "grepplus", shell_output("#{bin}/grepplus --help")
    assert_path_exists bin/"gp"
  end
end
