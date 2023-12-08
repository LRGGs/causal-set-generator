from flask import Flask, render_template, jsonify, request


def create_app(num_boxes=3):
    app = Flask(__name__)

    # Initialize an empty list of colored boxes
    colored_boxes = [{'id': i + 1, 'color': 'red'} for i in range(num_boxes)]

    @app.route('/')
    def index():
        return render_template('index.html', boxes=colored_boxes)

    @app.route('/api/boxes', methods=['GET'])
    def get_boxes():
        return jsonify(colored_boxes)

    @app.route('/api/box/<int:box_id>', methods=['GET', 'PUT'])
    def manage_box(box_id):
        if request.method == 'GET':
            box = next((box for box in colored_boxes if box['id'] == box_id), None)
            if box:
                return jsonify(box)
            else:
                return jsonify({'error': f'Box with id {box_id} not found'}), 404

        elif request.method == 'PUT':
            data = request.get_json()
            color = data.get('color')

            if not color:
                return jsonify({'error': 'Color parameter is required'}), 400

            box = next((box for box in colored_boxes if box['id'] == box_id), None)
            if box:
                box['color'] = color

                # Include a command to refresh the page in the response
                return jsonify({'status': 'success', 'refresh_page': True})
            else:
                return jsonify({'error': f'Box with id {box_id} not found'}), 404

    return app


if __name__ == '__main__':
    import sys

    # Get the number of boxes from the command-line arguments or default to 3
    num_boxes = int(sys.argv[1]) if len(sys.argv) > 1 else 3

    # Create the Flask app with the specified number of boxes
    app = create_app(num_boxes)

    # Run the Flask application
    app.run(debug=True)