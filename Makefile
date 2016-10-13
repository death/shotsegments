shotsegments: main.cpp
	g++ -o shotsegments `pkg-config --cflags opencv --libs opencv` main.cpp

clean:
	rm shotsegments
