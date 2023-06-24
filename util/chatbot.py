class Chatbot:
    def __init__(self, chatbotId, name, bot_type, payload=None):
        self.chatbotId = chatbotId
        self.name = name
        self.bot_type = bot_type
        self.payload = payload

    def validate_bot_type(self, bot_type):
        valid_types = ['freetalk', 'doctalk']
        if bot_type not in valid_types:
            raise ValueError(f"Invalid bot_type. Allowed values are: {valid_types}")
        return bot_type

    def greet(self):
        print(f"Hello, my name is {self.name}. How can I assist you today?")

    def prepare_data(self, file_paths):
        if self.bot_type != 'doctalk':
            return

        self.source_content = []
        for file_path in file_paths:
            with open(file_path, 'r') as file:
                content = file.read()
                self.source_content.append(content)

    def chat(self):
        while True:
            user_input = input("User: ")
            # Add your chatbot logic here to generate a response based on user_input
            response = self.generate_response(user_input)
            print(f"{self.name}: {response}")

            # Add an exit condition if desired
            if user_input.lower() == 'exit':
                break

    def generate_response(self, user_input):
        # Add your response generation logic here
        return "I am a chatbot. I can help you with various tasks."

# Example usage:
my_chatbot = Chatbot("12345", "MyChatBot", "freetalk")
my_chatbot.greet()
my_chatbot.chat()
