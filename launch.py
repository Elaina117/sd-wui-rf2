from modules import launch_utils

args = launch_utils.args
python = launch_utils.python
git = launch_utils.git
index_url = launch_utils.index_url
dir_repos = launch_utils.dir_repos

commit_hash = launch_utils.commit_hash
git_tag = launch_utils.git_tag

run = launch_utils.run
is_installed = launch_utils.is_installed
repo_dir = launch_utils.repo_dir

run_pip = launch_utils.run_pip
check_run_python = launch_utils.check_run_python
git_clone = launch_utils.git_clone
git_pull_recursive = launch_utils.git_pull_recursive
list_extensions = launch_utils.list_extensions
run_extension_installer = launch_utils.run_extension_installer
prepare_environment = launch_utils.prepare_environment
configure_for_tests = launch_utils.configure_for_tests
start = launch_utils.start


def main():
    if args.dump_sysinfo:
        filename = launch_utils.dump_sysinfo()

        print(f"Sysinfo saved as {filename}. Exiting...")

        exit(0)

    launch_utils.startup_timer.record("initial startup")

    with launch_utils.startup_timer.subcategory("prepare environment"):
        if not args.skip_prepare_environment:
            prepare_environment()

    # ======================================================================
    # TQDM Patch (JP & One-Line & Custom Bar) - Injected
    # ======================================================================
    try:
        import sys
        import tqdm

        # 強制一行表示用の display メソッド
        def jp_display(self, msg=None, pos=None):
            if self.disable: return
            if msg is None: msg = self.__str__()
            if len(msg) > 120: msg = msg[:115] + "..."
            # WebUIのログ制御を回避するため、システムの標準出力に直接書き込む
            fp = sys.__stdout__
            fp.write('\r' + msg + '      ')
            fp.flush()
            return True

        # 日本語化＆カスタムバー用の format_meter メソッド
        def jp_format_meter(n, total, elapsed, **kwargs):
            # --- デザイン設定 ---
            CHAR_FILL  = '█'
            CHAR_EMPTY = '░'
            BAR_LENGTH = 15
            # --------------------
            
            # prefix (Total progressなど)
            prefix = kwargs.get('prefix', '')
            initial = kwargs.get('initial', 0)
            rate = kwargs.get('rate')

            # 計算
            if total and total > 0:
                frac = float(n) / float(total)
                percentage = frac * 100
            else:
                frac = 0
                percentage = 0

            if rate is None and elapsed > 0: rate = (n - initial) / elapsed
            if rate and rate > 0:
                if rate < 1: rate_fmt = f"{rate:.2f}回/秒"
                else: inv_rate = 1 / rate; rate_fmt = f"{inv_rate:.2f}s/回"
            else: rate_fmt = "?s/回"

            if rate and rate > 0 and total:
                remaining_sec = (total - n) / rate
                remaining_str = tqdm.tqdm.format_interval(remaining_sec)
            else: remaining_str = "?"

            elapsed_str = tqdm.tqdm.format_interval(elapsed)
            
            fill_len = int(BAR_LENGTH * frac)
            if fill_len > BAR_LENGTH: fill_len = BAR_LENGTH
            empty_len = BAR_LENGTH - fill_len
            bar_str = CHAR_FILL * fill_len + CHAR_EMPTY * empty_len

            head = f"{prefix}: " if prefix else ""
            final_str = f"{head}[{bar_str}] {percentage:3.0f}% {n}/{total} [{elapsed_str} / 残り: {remaining_str}, {rate_fmt}]"
            return final_str

        # tqdmクラス本体のメソッドを直接上書きする
        tqdm.tqdm.format_meter = staticmethod(jp_format_meter)
        tqdm.tqdm.display = jp_display
        
        #print("✨ TQDM Patch Applied successfully inside launch.py!")
    except Exception as e:
        #print(f"⚠️ TQDM Patch failed to apply: {e}")
        pass
    # ======================================================================

    if args.test_server:
        configure_for_tests()

    if args.forge_ref_a1111_home:
        launch_utils.configure_forge_reference_checkout(args.forge_ref_a1111_home)

    start()


if __name__ == "__main__":
    main()