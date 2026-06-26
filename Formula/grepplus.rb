class Grepplus < Formula
  desc "Hybrid code search CLI — grep-fast by default, semantic when you need it"
  homepage "https://github.com/Mixpeal/grepplus"
  url "https://github.com/Mixpeal/grepplus/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "633f3f35240ee62a95ce00cb566b08cc32e3f11dd0d1ca5541b472f4a4118158"
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
