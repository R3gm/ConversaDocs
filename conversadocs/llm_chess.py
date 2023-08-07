import os
import chess
import chess.pgn

# credit code https://github.com/notnil/chess-gpt
def get_legal_moves(board):
    """Returns a list of legal moves in UCI notation."""
    return list(map(board.san, board.legal_moves))

def init_game() -> tuple[chess.pgn.Game, chess.Board]:
    """Initializes a new game."""
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["White"] = "User"
    game.headers["Black"] = "Chess-engine"
    del game.headers["Event"]
    del game.headers["Date"]
    del game.headers["Site"]
    del game.headers["Round"]
    del game.headers["Result"]
    game.setup(board)
    return game, board

def generate_prompt(game: chess.pgn.Game, board: chess.Board) -> str:

    moves = get_legal_moves(board)
    moves_str = ",".join(moves)
    return f"""
    The task is play a Chess game:
    You are the Chess-engine playing a chess match against the user as black and trying to win.

    The current FEN notation is:
    {board.fen()}

    The next valid moves are:
    {moves_str}

    Continue the game.
    {str(game)[:-2]}"""

def get_move(content, moves):
    lines = content.splitlines()
    for line in lines:
      for lm in moves:
        if lm in line:
          return lm

class ChessGame:
    def __init__(self, docschatllm):
      self.docschatllm = docschatllm

    def start_game(self):
        self.game, self.board = init_game()
        self.game_cp, _ = init_game()
        self.node = self.game
        self.node_copy = self.game_cp

        svg_board = chess.svg.board(self.board, size=350)
        return svg_board, "Valid moves: "+",".join(get_legal_moves(self.board)) # display(self.board)

    def user_move(self, move_input):
        try:
          self.board.push_san(move_input)
        except ValueError:
          print("Invalid move")
          svg_board = chess.svg.board(self.board, size=350)
          return svg_board, "Valid moves: "+",".join(get_legal_moves(self.board)), 'Invalid move'
        self.node = self.node.add_variation(self.board.move_stack[-1])
        self.node_copy = self.node_copy.add_variation(self.board.move_stack[-1])

        if self.board.is_game_over():
          svg_board = chess.svg.board(self.board, size=350)
          return svg_board, ",".join(get_legal_moves(self.board)), 'GAME OVER'

        prompt = generate_prompt(self.game, self.board)
        print("Prompt: \n"+prompt)
        print("#############")
        for i in range(10): #tries
          if i == 9:
            svg_board = chess.svg.board(self.board, size=350)
            return svg_board, ",".join(get_legal_moves(self.board)), "The model can't do a valid move"
          try:
            """Returns the move from the prompt."""
            content = self.docschatllm.llm.predict(prompt) ### from selected model ###
            #print(moves)
            print("Response: \n"+content)
            print("#############")

            moves = get_legal_moves(self.board)
            move = get_move(content, moves)
            print(move)
            print("#############")
            self.board.push_san(move)
            break
          except:
            prompt = prompt[1:]
            print("attempt a move.")
        self.node = self.node.add_variation(self.board.move_stack[-1])
        self.node_copy = self.node_copy.add_variation(self.board.move_stack[-1])
        svg_board = chess.svg.board(self.board, size=350)
        return svg_board, "Valid moves: "+",".join(get_legal_moves(self.board)), ''
